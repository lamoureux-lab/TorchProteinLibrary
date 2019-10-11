#include <Kernels.h>

__global__ void partialSum(float* coords, float* assigned_params, int* num_atoms, float* volume, 
                                int box_size, int coords_stride, float res){

    int d = 2;
    float rel_var = 1.0;
    int batch_index = threadIdx.x;
	float *single_volume = volume + batch_index * box_size*box_size*box_size;
	float *single_coords = coords + batch_index * coords_stride;
    float *single_params = assigned_params + batch_index * coords_stride/3;
	int n_atoms = num_atoms[batch_index];
	for(int idx = 0; idx<n_atoms; idx++){
        float sigma = single_params[idx] * rel_var;
		float x = single_coords[3*idx + 0],
			y = single_coords[3*idx + 1],
			z = single_coords[3*idx + 2];
		int x_i = floor(x/res);
		int y_i = floor(y/res);
		int z_i = floor(z/res);
		for(int i=x_i-d; i<=(x_i+d);i++){
			for(int j=y_i-d; j<=(y_i+d);j++){
				for(int k=z_i-d; k<=(z_i+d);k++){
					if( (i>=0 && i<box_size) && (j>=0 && j<box_size) && (k>=0 && k<box_size) ){
						int cell_idx = k + j*box_size + i*box_size*box_size;							
						float r2 = (x - i*res)*(x - i*res)+\
						(y - j*res)*(y - j*res)+\
						(z - k*res)*(z - k*res);
						single_volume[cell_idx] += exp(-r2/(sigma*sigma));
					}
				}
			}
		}
	}
}

void gpu_computeCoords2Volume(	float *coords,
                                float *assigned_params,
							    int *num_atoms, 
								float *volume,
								int batch_size,
                                int box_size,
                                int coords_stride,
								float res){
	partialSum<<<1, batch_size>>>(coords, assigned_params, num_atoms, volume, box_size, coords_stride, res);
}