#ifndef CPAIRS2FORCES_H_
#define CPAIRS2FORCES_H_
#include <TH.h>
#include <THC.h>

class cCoords2Volumes{

	THCState *state;

    THCudaIntTensor *input_types;
    
    int box_size, num_types;
    float resolution;
    	
public:

	cCoords2Volumes( int num_types, int box_size, float resolution );
	
    void computeForward(    THCState *state,
                            THCudaTensor *gpu_plain_coords, 
                            THCudaIntTensor *gpu_offsets, 
                            THCudaIntTensor *gpu_num_coords_of_type,
                            THCudaTensor *gpu_volume 
                            );
                        
	                        
	~cCoords2Volumes();

};

#endif