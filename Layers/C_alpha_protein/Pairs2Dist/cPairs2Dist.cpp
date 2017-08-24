#include <iostream>
#include <fstream>
#include <exception>
#include <limits>
#include <math.h>
#include <set>

#include "cPairs2Dist.h"
#include "../cTensorProteinCUDAKernels.h"
#include "../cPairwisePotentialsKernels.h"

cPairs2Dist::cPairs2Dist(   THCState *state, 
                            THCudaIntTensor *input_types,
                            int num_types,
                            int num_bins,
                            float resolution,
                            int angles_length,              // input actual number of angles in the sequence
                            int stride                      // stride in the output array (default angles_length)
                   ){
    this->angles_length = angles_length;
    this->stride = stride;
    this->num_types = num_types;
    this->num_bins = num_bins;
    this->resolution = resolution;

    this->input_types=THCudaIntTensor_new(state);
    THCudaIntTensor_set(state, this->input_types, input_types);
    
}
cPairs2Dist::~cPairs2Dist(){
    THCudaIntTensor_free(state, this->input_types);
}
void cPairs2Dist::computeForward(   THCudaTensor *input_pairs,     // input coords, 3 x maxlen + 1 
                                    THCudaTensor *output_dist      // output pair coords, 3 x maxlen + 1 x maxlen + 1
                                    ){
    
	// Forward pass pairwise coordinates -> forces
    cpu_computePairwiseDistributions(   THCudaTensor_data(state, input_pairs ),
                                        THCudaIntTensor_data(state, this->input_types ),
                                        THCudaTensor_data(state, output_dist),
                                        this->num_types,
                                        this->num_bins,
                                        this->resolution,
                                        this->angles_length, this->stride);
}   

void cPairs2Dist::computeBackward(  THCudaTensor *gradInput_pairs,   //output gradient of the input coords
                                    THCudaTensor *gradOutput_dist,    //input gradient of the pairwise coordinates
                                    THCudaTensor *input_pairs
                                    ){  

    cpu_backwardPairwiseDistributions(  THCudaTensor_data(state, gradInput_pairs ),
                                        THCudaTensor_data(state, gradOutput_dist ),
                                        THCudaIntTensor_data(state, this->input_types ),
                                        THCudaTensor_data(state, input_pairs ),
                                        this->num_types,
                                        this->num_bins,
                                        this->resolution,
                                        this->angles_length, this->stride);
}
