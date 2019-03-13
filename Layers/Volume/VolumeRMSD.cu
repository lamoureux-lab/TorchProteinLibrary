#include "VolumeRMSD.h"

__global__ void fillVolume( REAL *d_volume,  
							REAL add, 
							REAL T0_x, REAL T0_y, REAL T0_z,
							REAL D_x, REAL D_y, REAL D_z,
							float resolution){
		uint x = blockIdx.y;
		uint y = blockIdx.z;
		uint z = threadIdx.x;
		uint volume_size = gridDim.y;

		uint half_size = volume_size/2;
		uint vol2_bytes = volume_size*volume_size;
		
		REAL *vol = d_volume + x*vol2_bytes + y*volume_size;

		REAL T1_x = x*resolution, T1_y = y*resolution, T1_z = z*resolution;
		if (x >= half_size)
			T1_x = -(float(volume_size) - float(x)) * resolution;
		if (y >= half_size)
			T1_y = -(float(volume_size) - float(y)) * resolution;
		if (z >= half_size)
			T1_z = -(float(volume_size) - float(z)) * resolution;

		vol[z] = sqrt((T0_x - T1_x)*(T0_x - T1_x) + (T0_y - T1_y)*(T0_y - T1_y) + (T0_z - T1_z)*(T0_z - T1_z) + \
								(T0_x - T1_x)*D_x + (T0_y - T1_y)*D_y + (T0_z - T1_z)*D_z + add);

}

void gpu_VolumeRMSD(REAL *d_volume,  
					REAL add, 
					REAL T0_x, REAL T0_y, REAL T0_z,
					REAL D_x, REAL D_y, REAL D_z, 
					int volume_size, float resolution){
		
	dim3 dim_special(1, volume_size, volume_size);
	fillVolume<<<dim_special, volume_size>>>(d_volume, add, T0_x, T0_y, T0_z, D_x, D_y, D_z, resolution);
}


