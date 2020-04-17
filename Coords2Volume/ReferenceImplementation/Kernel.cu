#include "Kernel.h"
#include <stdio.h>
__global__ void projectToTensor(float* coords, int num_atoms, float *volume, int spatial_dim, float res){
	int d = 2;
	for(int atom_idx = 0; atom_idx<3*num_atoms; atom_idx+=3){
		float x = coords[atom_idx],
			y = coords[atom_idx + 1],
			z = coords[atom_idx + 2];
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

void gpu_coords2volume(	float *coords,
                        int num_atoms,
                        float *volume,
						int spatial_dim,
                        float res){
	projectToTensor<<<1, 1>>>(coords, num_atoms, volume, spatial_dim, res);
}




__global__ void projectToCell(float* coords, int num_atoms, float *volume, int spatial_dim, float res){
	int d = 2;
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;
	
    if( !((i>=0 && i<spatial_dim) && (j>=0 && j<spatial_dim) && (k>=0 && k<spatial_dim)) )
        return;
    
    uint idx = k + j*spatial_dim + i*spatial_dim*spatial_dim;
    float result = 0.0;
    for(int atom_idx = 0; atom_idx<num_atoms; atom_idx++){
		float x = coords[3*atom_idx], y = coords[3*atom_idx + 1], z = coords[3*atom_idx + 2];
		
        int x_i = floor(x/res);
		int y_i = floor(y/res);
		int z_i = floor(z/res);

        if(__sad(x_i, i, 0)>d) continue;
        if(__sad(y_i, j, 0)>d) continue;
        if(__sad(z_i, k, 0)>d) continue;

        float r2 = (x - i*res)*(x - i*res)+(y - j*res)*(y - j*res)+(z - k*res)*(z - k*res);

		result += exp(-r2/2.0);
	}
    
    volume[idx] = result;
}

void gpu_coords2volume_cell(float *coords,
                            int num_atoms,
                            float *volume,
						    int spatial_dim,
                            float res){
    dim3 threadsPerBlock(4, 4, 4);
    dim3 numBlocks( spatial_dim/threadsPerBlock.x + 1,
                    spatial_dim/threadsPerBlock.y + 1,
                    spatial_dim/threadsPerBlock.z + 1);
	projectToCell<<<numBlocks,threadsPerBlock>>>(coords, num_atoms, volume, spatial_dim, res);
}

