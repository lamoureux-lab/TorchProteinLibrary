#include <KernelsStress.h>
#include <stdio.h>


__global__ void projectToGrid(  float *coords, float *vectors, int num_atoms, float *volume, 
                                int box_size, float res){
    int d=2;
    int vol_index = threadIdx.x;
	float *single_volume = volume + vol_index * box_size*box_size*box_size;

    float cell_x, cell_y, cell_z;
    long cell_idx;

    float x, y, z;
    int x_i, y_i, z_i;
    float r2;


	for(int idx = 0; idx<num_atoms; idx++){
        float sigma2 = 1.0;
		x = coords[3*idx + 0];
		y = coords[3*idx + 1];
		z = coords[3*idx + 2];
		
        x_i = floor(x/res);
		y_i = floor(y/res);
		z_i = floor(z/res);
		for(int i=x_i-d; i<=(x_i+d);i++){
			for(int j=y_i-d; j<=(y_i+d);j++){
				for(int k=z_i-d; k<=(z_i+d);k++){
					if( (i>=0 && i<box_size) && (j>=0 && j<box_size) && (k>=0 && k<box_size) ){
						cell_idx = k + j*box_size + i*box_size*box_size;
						cell_x = i*res;
                        cell_y = j*res;
                        cell_z = k*res;
                        r2 = (x-cell_x)*(x-cell_x) + (y-cell_y)*(y-cell_y) + (z-cell_z)*(z-cell_z);
						single_volume[cell_idx] += vectors[3*idx+vol_index] * exp(-r2/sigma2);
					}
				}
			}
		}
	}
}


void gpu_projectToGrid(	float *coords,
                        float *vectors,
                        int num_atoms, 
                        float *volume,
                        int box_size,
                        float res){
	projectToGrid<<<1, 3>>>(coords, vectors, num_atoms, volume, box_size, res);
}