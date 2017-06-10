#include "cProteinLoader.h"
#include <iostream>
#include <string>
#include <THC/THC.h>
#include <TH/TH.h>
#include <math.h>


#define d 2
	__global__ void projectToTensor(float* d_flat_data, int* d_n_atoms, size_t* d_offsets, float *out, 
									int num_atom_types, int spatial_dim, float res){
		
        size_t func_index = threadIdx.x + blockIdx.x*blockDim.x;
		float *volume = out + func_index * spatial_dim*spatial_dim*spatial_dim;
		float *atoms_coords = d_flat_data + d_offsets[func_index];
		int n_atoms = d_n_atoms[func_index];
		for(int atom_idx = 0; atom_idx<n_atoms; atom_idx++){
			float 	x = atoms_coords[3*atom_idx],
					y = atoms_coords[3*atom_idx + 1],
					z = atoms_coords[3*atom_idx + 2];
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

    __global__ void projectFromTensor(  float* d_flat_data, int* d_n_atoms, size_t* d_offsets, float *grad,
                                        int num_atom_types, int spatial_dim, float res)
    /*
    Input:
        d_flat_data: coordinates in a flat array:
            d_flat_data: {atom_type1 .. atom_typeM}
            atom_type: {x1,y1,z1 .. xL,yL,zL}
        d_n_atoms: number of atoms in each atom_type 
        d_offsets: offset for coordinates for each atom_type volume
        grad: gradient to be projected on atoms
    Output: 
        d_flat_data: coordinates are rewritten for each atom to store the gradient projection
    */
    {
		size_t func_index = threadIdx.x + blockIdx.x*blockDim.x;
		float *volume = grad + func_index * spatial_dim*spatial_dim*spatial_dim;
		float *atoms_coords = d_flat_data + d_offsets[func_index];
		int n_atoms = d_n_atoms[func_index];
		for(int atom_idx = 0; atom_idx<n_atoms; atom_idx++){
			float 	x = atoms_coords[3*atom_idx],
					y = atoms_coords[3*atom_idx + 1],
					z = atoms_coords[3*atom_idx + 2];
            atoms_coords[3*atom_idx] = 0.0;
            atoms_coords[3*atom_idx+1] = 0.0;
            atoms_coords[3*atom_idx+2] = 0.0;
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
							atoms_coords[3*atom_idx] -= (x - i*res)*volume[idx]*exp(-r2/2.0);
                            atoms_coords[3*atom_idx + 1] -= (y - j*res)*volume[idx]*exp(-r2/2.0);
                            atoms_coords[3*atom_idx + 2] -= (z - k*res)*volume[idx]*exp(-r2/2.0);
						}
					}
				}
			}
		}
	}



extern "C"{
    int getNumberOfAtoms( const char* proteinPath, 
                        int assigner_type){
        /*
        Return number of atoms to which an atom type was assigned.
        */
        cProteinLoader pL;
		pL.loadPDB(proteinPath);
		if(pL.assignAtomTypes(assigner_type)<0){
			return -1;
		}
        int num_assigned = 0;
		for(int j=0; j<pL.atomType.size();j++){
            if(pL.atomType[j]>=0)
			    num_assigned+=1;
		}
        return num_assigned;
    }
    int prepareProtein( const char* proteinPath, 
                        float resolution, 
                        int assigner_type, 
                        int spatial_dim, 
                        bool center,
                        THFloatTensor *data_pointer, 
                        THIntTensor *n_atoms,
                        THIntTensor *flat_indexes){
        /*
        Prepares protein for projecting on the grid.
        Returns 
        data_pointer: array of all the flattened coordinates x,y,z in the order of atom types
        n_atoms: number of atoms of each type
        flat_indexes: indexes of atoms in the flattened format, corresponding to the real indexes in cProteinLoader
        */
		cProteinLoader pL;
		pL.loadPDB(proteinPath);
		if(pL.assignAtomTypes(assigner_type)<0){
			return -1;
		}
		
		pL.res = resolution;
        if(center){
            pL.computeBoundingBox();
            //placing center of the bbox to the origin
            pL.shiftProtein( -0.5*(pL.b0 + pL.b1) ); 
            // placing center of the protein to the center of the grid
            pL.shiftProtein( 0.5*cVector3(spatial_dim, spatial_dim, spatial_dim)*pL.res ); 
        }
        // std::cout<<n_atoms->nDimension<<"\n";

        // n_atoms = THIntTensor_newWithSize1d(pL.num_atom_types);
        std::vector<float> coords; // vector of plain coords of a particular atom type
        std::vector<int> indexes;
		for(int i=0; i<pL.num_atom_types; i++){
            THIntTensor_set1d(n_atoms, i, 0);
			for(int j=0; j<pL.atomType.size();j++){
				if(pL.atomType[j]==i){
					coords.push_back(pL.r[j].v[0]);
					coords.push_back(pL.r[j].v[1]);
					coords.push_back(pL.r[j].v[2]);
                    indexes.push_back(j);
                    int inatoms = THIntTensor_get1d(n_atoms, i);
                    THIntTensor_set1d(n_atoms, i, inatoms+1);
				}
			}
            std::cout<<"Type:"<<i<<" num atoms = "<<THIntTensor_get1d(n_atoms, i)<<"\n";
		}
        // data_pointer = THFloatTensor_newWithSize1d(coords.size());
        for(int i=0; i<coords.size(); i++){
            THFloatTensor_set1d(data_pointer, i, coords[i]);
        }
        // flat_indexes = THIntTensor_newWithSize1d(indexes.size());
        for(int i=0; i<indexes.size(); i++){
            THIntTensor_set1d(flat_indexes, i, indexes[i]);
        }
        std::cout<<"Loaded protein: \n natoms = "<<indexes.size()<<"\n ncoords = "<<coords.size()<<"\n";
		return 1;
	}
    int saveProtein(    const char* initProteinPath,
                        const char* outputProteinPath,
                        THFloatTensor *data_pointer, 
                        THIntTensor *flat_indexes,
                        int n_atoms,
                        int assigner_type){
        /*
        Saves protein in a file.
        Input:
        data_pointer: array of all the flattened coordinates x,y,z in the order of atom types
        n_atoms: number of atoms totally
        flat_indexes: indexes of atoms in the flattened format, corresponding to the real indexes in cProteinLoader
        */
		cProteinLoader pL;
		pL.loadPDB(initProteinPath);
        if(pL.assignAtomTypes(assigner_type)<0){
			return -1;
		}
        // std::cout<<"Saving protein: "<<initProteinPath<<std::endl;
		for(int i=0; i<n_atoms; i++){
            int j = THIntTensor_get1d(flat_indexes, i);
            // std::cout<<i<<"->"<<j<<" | "<<data_pointer->size[0]<<std::endl;
            pL.r[j].v[0] = THFloatTensor_get1d(data_pointer, 3*i);
            pL.r[j].v[1] = THFloatTensor_get1d(data_pointer, 3*i+1);
            pL.r[j].v[2] = THFloatTensor_get1d(data_pointer, 3*i+2);
            
		}
        pL.savePDB(outputProteinPath);
        // std::cout<<"Saved protein: \n natoms = "<<n_atoms<<"\n";
		return 1;
	}

    int protProjectToTensor(    THCState *state,
                                THCudaTensor *batch4D,
                                THFloatTensor *data_pointer, 
                                THIntTensor *n_atoms,
                                int spatial_dim,
                                float resolution){
		// std::cout<<"t1"<<std::endl;
        int num_atom_types = n_atoms->size[0];
 		size_t *offsets = new size_t[num_atom_types];
        for(int a_type_idx=0; a_type_idx<num_atom_types; a_type_idx++){
            
            if(a_type_idx>0)
                offsets[a_type_idx] = offsets[a_type_idx - 1] + 3*THIntTensor_get1d(n_atoms, a_type_idx-1);
            else
                offsets[a_type_idx] = 0;

            // std::cout<<a_type_idx<<" "<<offsets[a_type_idx]<<"\n";
        }
        
		//data_array copy to gpu
		float *d_flat_data;
		int *d_n_atoms;
		size_t *d_offsets;
        cudaMalloc( (void**) &d_flat_data, (data_pointer->size[0])*sizeof(float) );
        cudaMalloc( (void**) &d_n_atoms, num_atom_types*sizeof(int) );
		cudaMalloc( (void**) &d_offsets, num_atom_types*sizeof(size_t) );

        cudaMemcpy( d_flat_data, THFloatTensor_data(data_pointer), (data_pointer->size[0])*sizeof(float), cudaMemcpyHostToDevice);		
        cudaMemcpy( d_n_atoms, THIntTensor_data(n_atoms), num_atom_types*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy( d_offsets, offsets, num_atom_types*sizeof(size_t), cudaMemcpyHostToDevice);
        
        // for(int i=0;i<THIntTensor_get1d(n_atoms,1)*3;i++){		
        //     std::cout<<THFloatTensor_get1d(data_pointer,i+offsets[1])<<"\n";
		// }

		float* grid = THCudaTensor_data(state, batch4D);
		projectToTensor<<<1, num_atom_types>>>(	d_flat_data, d_n_atoms, d_offsets,
												grid, 
                                                num_atom_types, 
                                                spatial_dim,
												resolution);		
		delete[] offsets;
		cudaFree(d_n_atoms);
		cudaFree(d_flat_data);
		cudaFree(d_offsets);
		
		return 1;
	}

    int protProjectFromTensor(  THCState *state,
                                THCudaTensor *gradient4D,
                                THFloatTensor *data_pointer,
                                THFloatTensor *gradients_data_pointer,
                                THIntTensor *n_atoms,
                                int spatial_dim,
                                float resolution){
		// std::cout<<"ft1"<<std::endl;
        int num_atom_types = n_atoms->size[0];
 		size_t *offsets = new size_t[num_atom_types];
        for(int a_type_idx=0; a_type_idx<num_atom_types; a_type_idx++){
            if(a_type_idx>0)
                offsets[a_type_idx] = offsets[a_type_idx - 1] + 3*THIntTensor_get1d(n_atoms, a_type_idx-1);
            else
                offsets[a_type_idx] = 0;
        }
        // std::cout<<"ft2"<<std::endl;
		//data_array copy to gpu
		float *d_flat_data;
		int *d_n_atoms;
		size_t *d_offsets;
        cudaMalloc( (void**) &d_flat_data, (data_pointer->size[0])*sizeof(float) );
        cudaMalloc( (void**) &d_n_atoms, num_atom_types*sizeof(int) );
		cudaMalloc( (void**) &d_offsets, num_atom_types*sizeof(size_t) );

        cudaMemcpy( d_flat_data, THFloatTensor_data(data_pointer), (data_pointer->size[0])*sizeof(float), cudaMemcpyHostToDevice);		
        cudaMemcpy( d_n_atoms, THIntTensor_data(n_atoms), num_atom_types*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy( d_offsets, offsets, num_atom_types*sizeof(size_t), cudaMemcpyHostToDevice);
        
		// std::cout<<"ft3"<<std::endl;
		float* grid = THCudaTensor_data(state, gradient4D);
		projectFromTensor<<<1, num_atom_types>>>(	d_flat_data, d_n_atoms, d_offsets,
                                                    grid, 
                                                    num_atom_types, 
                                                    spatial_dim,
                                                    resolution);	
        // std::cout<<"ft4"<<std::endl;
        cudaMemcpy( THFloatTensor_data(gradients_data_pointer), d_flat_data, (data_pointer->size[0])*sizeof(float), cudaMemcpyDeviceToHost);

		delete[] offsets;
		cudaFree(d_n_atoms);
		cudaFree(d_flat_data);
		cudaFree(d_offsets);
		// std::cout<<"ft5"<<std::endl;
		return 1;
	}
}