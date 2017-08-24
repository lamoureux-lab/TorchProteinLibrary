#include <THC/THC.h>
#include "cAngles2CoordsDihedral.h"
#include <iostream>

extern THCState *state;
float R = 3.8;
extern "C" {
    int Angles2Coords_forward(  THCudaTensor *input_angles, 
                                THCudaTensor *output_coords, 
                                THIntTensor *angles_length, 
                                THCudaTensor *A
                            ){
        if(input_angles->nDimension == 2){
            cAngles2CoordsDihedral a2c(state, A, input_angles, NULL, R, THIntTensor_get1d(angles_length, 0));
            a2c.computeForward(input_angles, output_coords);
        }else{
            // #pragma omp parallel for num_threads(10)
            for(int i=0; i<input_angles->size[0]; i++){
                THCudaTensor *single_input_angles, *single_A, *single_output_coords;
                single_input_angles = THCudaTensor_new(state);
	            single_A = THCudaTensor_new(state);
                single_output_coords = THCudaTensor_new(state);

                THCudaTensor_select(state, single_input_angles, input_angles, 0, i);
                THCudaTensor_select(state, single_A, A, 0, i);
                THCudaTensor_select(state, single_output_coords, output_coords, 0, i);
                cAngles2CoordsDihedral a2c(state, single_A, single_input_angles, NULL, R, THIntTensor_get1d(angles_length, i));
                a2c.computeForward(single_input_angles, single_output_coords);

                THCudaTensor_free(state, single_input_angles);
                THCudaTensor_free(state, single_A);
                THCudaTensor_free(state, single_output_coords);
            }
        }
    }
    int Angles2Coords_backward(     THCudaTensor *gradInput,
                                    THCudaTensor *gradOutput,
                                    THCudaTensor *input_angles, 
                                    THIntTensor *angles_length, 
                                    THCudaTensor *A,   
                                    THCudaTensor *dr_dangle
                            ){
        if(input_angles->nDimension == 2){
            cAngles2CoordsDihedral a2c(state, A, input_angles, dr_dangle, R, THIntTensor_get1d(angles_length, 0));
            a2c.computeBackward(gradInput, gradOutput);
        }else{
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<input_angles->size[0]; i++){
                THCudaTensor *single_input_angles, *single_A, *single_dr_dangle, *single_gradInput, *single_gradOutput;
                single_input_angles = THCudaTensor_new(state);
	            single_A = THCudaTensor_new(state);
                single_dr_dangle = THCudaTensor_new(state);
                single_gradInput = THCudaTensor_new(state);
                single_gradOutput = THCudaTensor_new(state);

                THCudaTensor_select(state, single_input_angles, input_angles, 0, i);
                THCudaTensor_select(state, single_A, A, 0, i);
                THCudaTensor_select(state, single_dr_dangle, dr_dangle, 0, i);
                THCudaTensor_select(state, single_gradInput, gradInput, 0, i);
                THCudaTensor_select(state, single_gradOutput, gradOutput, 0, i);
                
                cAngles2CoordsDihedral a2c(state, single_A, single_input_angles, single_dr_dangle, R, THIntTensor_get1d(angles_length, i));
                a2c.computeBackward(single_gradInput, single_gradOutput);

                THCudaTensor_free(state, single_input_angles);
                THCudaTensor_free(state, single_A);
                THCudaTensor_free(state, single_dr_dangle);
                THCudaTensor_free(state, single_gradInput);
                THCudaTensor_free(state, single_gradOutput);
            }
        }
    }


}