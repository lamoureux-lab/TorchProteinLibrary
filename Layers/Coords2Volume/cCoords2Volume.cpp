#include <iostream>
#include <fstream>
#include <exception>
#include <limits>
#include <math.h>
#include <set>

#include "cCoords2Volume.h"
#include "cCoords2Volume_CUDAKernels.h"

cCoords2Volume::cCoords2Volume(		int num_types,
										int box_size,
										float resolution
								){
	this->num_types = num_types;
	this->box_size = box_size;
	this->resolution = resolution;
}

cCoords2Volume::~cCoords2Volume(){
}

void cCoords2Volume::computeForward(   THCState *state,
										THCudaTensor *gpu_plain_coords, 
										THCudaIntTensor *gpu_offsets, 
										THCudaIntTensor *gpu_num_coords_of_type,
										THCudaTensor *gpu_volume
                            		){
	THCudaTensor_zero(state, gpu_volume);
	gpu_computeCoords2Volume(    THCudaTensor_data(state, gpu_plain_coords ),
									THCudaIntTensor_data(state, gpu_offsets ),
									THCudaIntTensor_data(state, gpu_num_coords_of_type),
									THCudaTensor_data(state, gpu_volume),
									this->num_types,
									this->box_size,
									this->resolution);

}