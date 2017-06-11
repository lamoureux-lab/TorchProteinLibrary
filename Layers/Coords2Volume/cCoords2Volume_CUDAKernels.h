#include <THC/THC.h>

void gpu_computeCoords2Volume(    float *gpu_plain_coords,
							    int *gpu_offsets, 
								int *gpu_num_coords_of_type,
								float *gpu_volume,
								int num_types,
								int box_size,
								float resolution);