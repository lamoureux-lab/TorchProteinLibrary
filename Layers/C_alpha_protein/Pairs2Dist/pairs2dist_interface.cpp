#include <THC/THC.h>
#include "cPairs2Dist.h"
#include <iostream>

extern THCState *state;
float R = 3.8;
extern "C" {
    int Pairs2Dist_forward(     THCudaTensor *input_pairs, 
                                THCudaTensor *output_dist,
                                THCudaIntTensor *input_types, 
                                THIntTensor *angles_length,
                                int num_types,
                                int num_bins,
                                float resolution
                            ){
        if(input_pairs->nDimension == 1){
            int max_angles = sqrt(input_pairs->size[0]/3)-1;
            cPairs2Dist p2d(state, input_types, num_types, num_bins, resolution, THIntTensor_get1d(angles_length, 0), max_angles);
            p2d.computeForward(input_pairs, output_dist);
        }else{
            int max_angles = sqrt(input_pairs->size[1]/3)-1;
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<input_pairs->size[0]; i++){
                THCudaTensor *single_input_pairs, *single_output_dist;
                THCudaIntTensor *single_input_types;
                single_input_pairs = THCudaTensor_new(state);
                single_output_dist = THCudaTensor_new(state);
                single_input_types = THCudaIntTensor_new(state);

                THCudaTensor_select(state, single_input_pairs, input_pairs, 0, i);
                THCudaTensor_select(state, single_output_dist, output_dist, 0, i);
                THCudaIntTensor_select(state, single_input_types, input_types, 0, i);
                cPairs2Dist p2d(state, single_input_types, num_types, num_bins, resolution, THIntTensor_get1d(angles_length, i), max_angles);
                p2d.computeForward(single_input_pairs, single_output_dist);

                THCudaTensor_free(state, single_input_pairs);
                THCudaTensor_free(state, single_output_dist);
                THCudaIntTensor_free(state, single_input_types);
            }
        }
    }
    int Pairs2Dist_backward(THCudaTensor *gradInput_pairs,
                            THCudaTensor *gradOutput_dist,
                            THCudaTensor *input_pairs, 
                            THCudaIntTensor *input_types, 
                            THIntTensor *angles_length,
                            int num_types,
                            int num_bins,
                            float resolution
                            ){
        if(input_pairs->nDimension == 1){
            int max_angles = sqrt(gradInput_pairs->size[0]/3) - 1;
            cPairs2Dist p2d(state, input_types, num_types, num_bins, resolution, THIntTensor_get1d(angles_length, 0), max_angles);
            p2d.computeBackward(gradInput_pairs, gradOutput_dist, input_pairs);
        }else{
            int max_angles = sqrt(gradInput_pairs->size[1]/3) - 1;
            // #pragma omp parallel for num_threads(10)
            for(int i=0; i<input_pairs->size[0]; i++){
                THCudaTensor *single_gradInput_pairs, *single_gradOutput_dist, *single_input_pairs;
                THCudaIntTensor  *single_input_types;
                single_gradInput_pairs = THCudaTensor_new(state);
                single_gradOutput_dist = THCudaTensor_new(state);
                single_input_types = THCudaIntTensor_new(state);
                single_input_pairs = THCudaTensor_new(state);

                THCudaTensor_select(state, single_gradInput_pairs, gradInput_pairs, 0, i);
                THCudaTensor_select(state, single_gradOutput_dist, gradOutput_dist, 0, i);
                THCudaIntTensor_select(state, single_input_types, input_types, 0, i);
                THCudaTensor_select(state, single_input_pairs, input_pairs, 0, i);
                
                cPairs2Dist p2d(state, single_input_types, num_types, num_bins, resolution, THIntTensor_get1d(angles_length, i), max_angles);
                p2d.computeBackward(single_gradInput_pairs, single_gradOutput_dist, single_input_pairs);

                THCudaTensor_free(state, single_input_pairs);
                THCudaTensor_free(state, single_gradInput_pairs);
                THCudaIntTensor_free(state, single_input_types);
                THCudaTensor_free(state, single_gradOutput_dist);
            }
        }
    }
}