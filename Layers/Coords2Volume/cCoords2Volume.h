#ifndef CCOORDS2VOLUME_H_
#define CCOORDS2VOLUME_H_
#include <TH.h>
#include <THC.h>

class cCoords2Volume{

	THCState *state;

    THCudaIntTensor *input_types;
    
    int box_size, num_types;
    float resolution;
    	
public:

	cCoords2Volume( int num_types, int box_size, float resolution );
	
    void computeForward(    THCState *state,
                            THCudaTensor *gpu_plain_coords, 
                            THCudaIntTensor *gpu_offsets, 
                            THCudaIntTensor *gpu_num_coords_of_type,
                            THCudaTensor *gpu_volume 
                            );
                        
	                        
	~cCoords2Volume();

};

#endif