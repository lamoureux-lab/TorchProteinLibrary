#include "cCoords2Volume_CUDAKernels.h"


__global__ void projectToTensor(    float* d_flat_data, int* d_n_atoms, int* d_offsets, float *out, 
									int num_atom_types, int spatial_dim,
									float res){
		int d = 2;
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



void gpu_computeCoords2Volume(		float *gpu_plain_coords,
							    int *gpu_offsets, 
								int *gpu_num_coords_of_type,
								float *gpu_volume,
								int num_types,
								int box_size,
								float resolution){

	projectToTensor<<<1, num_types>>>(	gpu_plain_coords, gpu_num_coords_of_type, gpu_offsets,
											gpu_volume, num_types, box_size, resolution);

}