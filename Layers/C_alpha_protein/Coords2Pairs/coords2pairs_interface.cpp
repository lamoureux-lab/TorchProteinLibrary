#include <THC/THC.h>
#include "cCoords2Pairs.h"
#include <iostream>

extern THCState *state;
float R = 3.8;
extern "C" {
    int Coords2Pairs_forward(  THCudaTensor *input_coords, 
                                THCudaTensor *output_pairs, 
                                THIntTensor *angles_length 
                            ){
        if(input_coords->nDimension == 1){
            cCoords2Pairs c2p(state, THIntTensor_get1d(angles_length, 0), input_coords->size[0]/3 - 1);
            c2p.computeForward(input_coords, output_pairs);
        }else{
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<input_coords->size[0]; i++){
                THCudaTensor *single_input_coords, *single_output_pairs;
                single_input_coords = THCudaTensor_new(state);
                single_output_pairs = THCudaTensor_new(state);

                THCudaTensor_select(state, single_input_coords, input_coords, 0, i);
                THCudaTensor_select(state, single_output_pairs, output_pairs, 0, i);
                cCoords2Pairs c2p(state, THIntTensor_get1d(angles_length, i), input_coords->size[1]/3 - 1);
                c2p.computeForward(single_input_coords, single_output_pairs);

                THCudaTensor_free(state, single_input_coords);
                THCudaTensor_free(state, single_output_pairs);
            }
        }
    }
    int Coords2Pairs_backward(      THCudaTensor *gradInput,
                                    THCudaTensor *gradOutput,
                                    THIntTensor *angles_length
                            ){
        if(gradInput->nDimension == 1){
            cCoords2Pairs c2p(state, THIntTensor_get1d(angles_length, 0), gradInput->size[0]/3 - 1);
            c2p.computeBackward(gradInput, gradOutput);
        }else{
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<gradInput->size[0]; i++){
                THCudaTensor *single_gradInput, *single_gradOutput;
                single_gradInput = THCudaTensor_new(state);
                single_gradOutput = THCudaTensor_new(state);

                THCudaTensor_select(state, single_gradInput, gradInput, 0, i);
                THCudaTensor_select(state, single_gradOutput, gradOutput, 0, i);
                
                cCoords2Pairs c2p(state, THIntTensor_get1d(angles_length, i), gradInput->size[1]/3 - 1);
                c2p.computeBackward(single_gradInput, single_gradOutput);

                THCudaTensor_free(state, single_gradInput);
                THCudaTensor_free(state, single_gradOutput);
            }
        }
    }
}