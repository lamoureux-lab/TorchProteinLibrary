#include "RotateGrid.h"

__global__ void gpuRotatePoint(REAL *d_rotations, REAL *d_grid, int batch_size, int volume_size){
    int batch_id = blockIdx.x;
    float *d_m = d_rotations + batch_id * 9;
    
    int volume = (volume_size*volume_size*volume_size);
    int volume_slice = (volume_size*volume_size);
        
    int x = blockIdx.y;
    int y = blockIdx.z;
    int z = threadIdx.x;

    float d_v[3] = { 2.0*float(x + 0.5 - volume_size/2.0)/ float(volume_size),
                     2.0*float(y + 0.5 - volume_size/2.0)/ float(volume_size),
                     2.0*float(z + 0.5 - volume_size/2.0)/ float(volume_size) };
    
    float *dst = d_grid + batch_id*volume*3 + x*volume_slice*3 + y*volume_size*3 + z*3;
    
    // x, y, z
    // dst[0] = d_m[0]*d_v[0]+d_m[1]*d_v[1]+d_m[2]*d_v[2];
	// dst[1] = d_m[3]*d_v[0]+d_m[4]*d_v[1]+d_m[5]*d_v[2];
	// dst[2] = d_m[6]*d_v[0]+d_m[7]*d_v[1]+d_m[8]*d_v[2];

    // z, y, x
    dst[2] = d_m[0]*d_v[0]+d_m[1]*d_v[1]+d_m[2]*d_v[2];
	dst[1] = d_m[3]*d_v[0]+d_m[4]*d_v[1]+d_m[5]*d_v[2];
	dst[0] = d_m[6]*d_v[0]+d_m[7]*d_v[1]+d_m[8]*d_v[2];

}

void cpu_RotateGrid(REAL *d_rotations, REAL *d_grid, int batch_size, int volume_size){

    dim3 dim_special(batch_size, volume_size, volume_size);
	gpuRotatePoint<<<dim_special, volume_size>>>(d_rotations, d_grid, batch_size, volume_size);

}