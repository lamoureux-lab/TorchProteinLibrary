#include "cProteinLoader.h"
#include <iostream>
#include <string>
#include <THC/THC.h>
#include <math.h>

#define d 2
	__global__ void projectToTensor(float* d_flat_data, int* d_n_atoms, size_t* d_offsets, float *out, 
									int batch_size, int num_atom_types, int spatial_dim,
									float res){
		size_t func_index = threadIdx.x + blockIdx.x*blockDim.x;
		float *volume = out + func_index * spatial_dim*spatial_dim*spatial_dim;
		float *atoms_coords = d_flat_data + d_offsets[func_index];
		int n_atoms = d_n_atoms[func_index];
		for(int atom_idx = 0; atom_idx<n_atoms; atom_idx+=3){
			float 	x = atoms_coords[atom_idx],
					y = atoms_coords[atom_idx + 1],
					z = atoms_coords[atom_idx + 2];
			int x_i = floor(x/res);
			int y_i = floor(y/res);
			int z_i = floor(z/res);
			for(int i=x_i-d; i<=(x_i+d);i++){
				for(int j=y_i-d; j<=(y_i+d);j++){
					for(int k=z_i-d; k<=(z_i+d);k++){
						if( (i>=0 && i<spatial_dim) && (j>=0 && j<spatial_dim) && (k>=0 && k<spatial_dim) ){
							int idx = k + j*spatial_dim + i*spatial_dim*spatial_dim;							
							float r2 = (x - i*res)*(x - i*res)+\
							(y - j*res)*(y - j*res)+\
							(z - k*res)*(z - k*res);
							volume[idx]+=exp(-r2/2.0);
						}
					}
				}
			}
		}
	}

extern "C"{
	int loadProtein(const char* proteinPath, 
					bool shift, bool rot, float resolution, 
					int assigner_type, int spatial_dim, THGenerator *gen, 
					float** data_pointer, int* n_atoms){

		cProteinLoader pL;
		pL.loadPDB(proteinPath);
		if(pL.assignAtomTypes(assigner_type)<0){
			return -1;
		}
		
		pL.res = resolution;
		pL.computeBoundingBox();
		//placing center of the bbox to the origin
		pL.shiftProtein( -0.5*(pL.b0 + pL.b1) ); 
		if(rot){
			// float alpha = THRandom_uniform(gen,0,2.0*M_PI);
 			// float beta = THRandom_uniform(gen,0,2.0*M_PI);
 			// float theta = THRandom_uniform(gen,0,2.0*M_PI);
 			// cMatrix33 random_rotation = cMatrix33::rotationXYZ(alpha,beta,theta);
			float u1 = THRandom_uniform(gen,0,1.0);
			float u2 = THRandom_uniform(gen,0,1.0);
			float u3 = THRandom_uniform(gen,0,1.0);
			float q[4];
			q[0] = sqrt(1-u1) * sin(2.0*M_PI*u2);
			q[1] = sqrt(1-u1) * cos(2.0*M_PI*u2);
			q[2] = sqrt(u1) * sin(2.0*M_PI*u3);
			q[3] = sqrt(u1) * cos(2.0*M_PI*u3);
			cMatrix33 random_rotation;
			random_rotation.m[0][0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
			random_rotation.m[0][1] = 2.0*(q[1]*q[2] - q[0]*q[3]);
			random_rotation.m[0][2] = 2.0*(q[1]*q[3] + q[0]*q[2]);

			random_rotation.m[1][0] = 2.0*(q[1]*q[2] + q[0]*q[3]);
			random_rotation.m[1][1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
			random_rotation.m[1][2] = 2.0*(q[2]*q[3] - q[0]*q[1]);

			random_rotation.m[2][0] = 2.0*(q[1]*q[3] - q[0]*q[2]);
			random_rotation.m[2][1] = 2.0*(q[2]*q[3] + q[0]*q[1]);
			random_rotation.m[2][2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
			pL.rotateProtein(random_rotation);
		}
		if(shift){
			float dx_max = fmax(0, spatial_dim*pL.res/2.0 - (pL.b1[0]-pL.b0[0])/2.0)*0.5;
			float dy_max = fmax(0, spatial_dim*pL.res/2.0 - (pL.b1[1]-pL.b0[1])/2.0)*0.5;
			float dz_max = fmax(0, spatial_dim*pL.res/2.0 - (pL.b1[2]-pL.b0[2])/2.0)*0.5;
			float dx = THRandom_uniform(gen,-dx_max,dx_max);
		 	float dy = THRandom_uniform(gen,-dy_max,dy_max);
		 	float dz = THRandom_uniform(gen,-dz_max,dz_max);
		 	pL.shiftProtein(cVector3(dx,dy,dz));
		}
		// placing center of the protein to the center of the grid
		pL.shiftProtein( 0.5*cVector3(spatial_dim, spatial_dim, spatial_dim)*pL.res ); 

		for(int i=0; i<pL.num_atom_types; i++){
			std::vector<float> coords; // vector of plain coords of a particular atom type
			for(int j=0; j<pL.atomType.size();j++){
				if(pL.atomType[j]==i){
					coords.push_back(pL.r[j].v[0]);
					coords.push_back(pL.r[j].v[1]);
					coords.push_back(pL.r[j].v[2]);
				}
			}
			float *coords_plain = new float[coords.size()];
			data_pointer[i] = coords_plain;
			n_atoms[i] = coords.size();
			for(int j=0; j<coords.size();j++)
				coords_plain[j]=coords[j];
		}
		return 1;
	}

	typedef struct{
		char **strings;
		size_t len;
		size_t ind;
	} batchInfo;

	batchInfo* createBatchInfo(int batch_size){
		//std::cout<<"Creating batch info of size = "<<batch_size<<"\n";
		batchInfo *binfo;
		binfo = new batchInfo;
		binfo->strings = new char*[batch_size];
		binfo->len = batch_size;
		binfo->ind = 0;
		return binfo;
	}

	void deleteBatchInfo(batchInfo* binfo){
		for(int i=0;i<binfo->len;i++){
			delete [] binfo->strings[i];
		}
		binfo->len=0;
		binfo->ind=0;
		delete binfo;
	}

	void pushProteinToBatchInfo(const char* filename, batchInfo* binfo){
		std::string str(filename);
		//std::cout<<"Pushing the string "<<str<<" to the position "<<pos<<"\n";
		//std::cout<<grid4D->nDimension<<"\n";

		binfo->strings[binfo->ind] = new char[str.length()+1];
		for(int i=0; i<str.length(); i++){
			binfo->strings[binfo->ind][i] = str[i];
		}
		binfo->strings[binfo->ind][str.length()]='\0';
		binfo->ind += 1;
		//std::cout<<binfo->grids4D[pos]->nDimension<<"\n";
	}

	void printBatchInfo(batchInfo* binfo){
		for(int i=0;i<min(binfo->len,binfo->ind);i++){
			std::cout<<binfo->strings[i]<<"\n";
		}
	}
	

	int loadProteinCUDA(THCState *state,
						 batchInfo* batch, THCudaTensor *batch5D,
						 bool shift, bool rot, float resolution,
						 int assigner_type, int spatial_dim){
		// std::cout<<"Flag!"<<std::endl;
		THGenerator *gen = THGenerator_new();
 		THRandom_seed(gen);
 		int num_atom_types;
 		if(assigner_type==1)num_atom_types=4;
 		else num_atom_types=11;
		// std::cout<<"Launched function"<<(batch->len)*num_atom_types<<std::endl;
 		float **data_array = new float*[(batch->len)*num_atom_types];
 		int *n_atoms = new int[(batch->len)*num_atom_types];
 		size_t *offsets = new size_t[(batch->len)*num_atom_types];
 		
 		std::vector<int> flags;
 		flags.resize(batch->len);
		#pragma omp parallel for num_threads(10)
		for(int i=0; i<batch->len; i++){
			int res = loadProtein(batch->strings[i], shift, rot, resolution, assigner_type, spatial_dim, gen, 
				data_array + i*num_atom_types, n_atoms + i*num_atom_types);
			flags[i] = res;
		}
		
		for(int i=0; i<batch->len; i++){
			if(flags[i]<0){
				std::cout<<"Corrupt file detected\n";
				for(int j=0;j<batch->len;j++){
					if(flags[j]>0)
						for(int k=0;k<num_atom_types;k++)
							delete[] data_array[k+j*num_atom_types];	
				}
				delete[] data_array;
				delete[] n_atoms;
				delete[] offsets;
				THGenerator_free(gen);
				return -1;
			}
		}

		size_t total_size = 0;
		for(int batch_idx=0; batch_idx<batch->len; batch_idx++){
			for(int a_type_idx=0; a_type_idx<num_atom_types; a_type_idx++){
				int volume_idx = a_type_idx + batch_idx*num_atom_types;
				if(volume_idx>0)
					offsets[volume_idx] = offsets[volume_idx - 1] + n_atoms[volume_idx-1];
				else
					offsets[volume_idx] = 0;
				total_size += n_atoms[volume_idx];
			}
		}

		// for(int i=0;i<batch->len;i++){
		// 	for(int j=0;j<num_atom_types;j++){
		// 		std::cout<<"("<<n_atoms[j+i*num_atom_types]<<","<<offsets[j+i*num_atom_types]<<"), ";
		// 	}
		// 	std::cout<<"\n";
		// }

		// for(int i=0;i<n_atoms[1];i++){
		// 	std::cout<<data_array[1][i]<<"\n";
		// }

		//data_array copy to gpu
		float *d_flat_data;
		int *d_n_atoms;
		size_t *d_offsets;
		cudaMalloc( (void**) &d_flat_data, total_size*sizeof(float) );
		cudaMalloc( (void**) &d_n_atoms, (batch->len)*num_atom_types*sizeof(int) );
		cudaMalloc( (void**) &d_offsets, (batch->len)*num_atom_types*sizeof(size_t) );
				
		cudaMemcpy( d_n_atoms, n_atoms, (batch->len)*num_atom_types*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy( d_offsets, offsets, (batch->len)*num_atom_types*sizeof(size_t), cudaMemcpyHostToDevice);
		
		for(int i=0; i<batch->len; i++){
			for(int j=0; j<num_atom_types; j++){
				int volume_idx = j+i*num_atom_types;
				cudaMemcpy( d_flat_data + offsets[volume_idx], data_array[j+i*num_atom_types], 
							n_atoms[j+i*num_atom_types]*sizeof(float), cudaMemcpyHostToDevice);
			}
		}
		float* grid = THCudaTensor_data(state, batch5D);
		projectToTensor<<<(batch->len), num_atom_types>>>(	d_flat_data, d_n_atoms, d_offsets,
															grid, 
															batch->len, num_atom_types, spatial_dim,
															resolution);
	
				
		for(int i=0;i<batch->len*num_atom_types;i++){
			delete[] data_array[i];
		}
		delete[] data_array;
		delete[] n_atoms;
		delete[] offsets;
		cudaFree(d_n_atoms);
		cudaFree(d_flat_data);
		cudaFree(d_offsets);
		THGenerator_free(gen);
		return 1;
	}
}