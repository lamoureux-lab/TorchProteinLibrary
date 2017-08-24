#include <THC/THC.h>
#include "cAngles2BasisDihedral.h"
#include <iostream>

extern THCState *state;
float R = 3.8;
extern "C" {
    int Angles2Basis_forward(  THCudaTensor *input_angles, 
                                THCudaTensor *output_basis, 
                                THIntTensor *angles_length, 
                                THCudaTensor *B
                            ){
        if(input_angles->nDimension == 2){
            cAngles2BasisDihedral a2b(state, B, input_angles, NULL, R, THIntTensor_get1d(angles_length, 0));
            a2b.computeForward(input_angles, output_basis);
        }else{
            // #pragma omp parallel for num_threads(10)
            for(int i=0; i<input_angles->size[0]; i++){
                THCudaTensor *single_input_angles, *single_B, *single_output_basis;
                single_input_angles = THCudaTensor_new(state);
	            single_B = THCudaTensor_new(state);
                single_output_basis = THCudaTensor_new(state);

                THCudaTensor_select(state, single_input_angles, input_angles, 0, i);
                THCudaTensor_select(state, single_B, B, 0, i);
                THCudaTensor_select(state, single_output_basis, output_basis, 0, i);
                cAngles2BasisDihedral a2b(state, single_B, single_input_angles, NULL, R, THIntTensor_get1d(angles_length, i));
                a2b.computeForward(single_input_angles, single_output_basis);

                THCudaTensor_free(state, single_input_angles);
                THCudaTensor_free(state, single_B);
                THCudaTensor_free(state, single_output_basis);
            }
        }
    }
    int Angles2Basis_backward(     THCudaTensor *gradInput,
                                    THCudaTensor *gradOutput,
                                    THCudaTensor *input_angles, 
                                    THIntTensor *angles_length, 
                                    THCudaTensor *B,   
                                    THCudaTensor *daxis_dangle
                            ){
        if(input_angles->nDimension == 2){
            cAngles2BasisDihedral a2b(state, B, input_angles, daxis_dangle, R, THIntTensor_get1d(angles_length, 0));
            a2b.computeBackward(gradInput, gradOutput);
        }else{
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<input_angles->size[0]; i++){
                THCudaTensor *single_input_angles, *single_B, *single_daxis_dangle, *single_gradInput, *single_gradOutput;
                single_input_angles = THCudaTensor_new(state);
	            single_B = THCudaTensor_new(state);
                single_daxis_dangle = THCudaTensor_new(state);
                single_gradInput = THCudaTensor_new(state);
                single_gradOutput = THCudaTensor_new(state);

                THCudaTensor_select(state, single_input_angles, input_angles, 0, i);
                THCudaTensor_select(state, single_B, B, 0, i);
                THCudaTensor_select(state, single_daxis_dangle, daxis_dangle, 0, i);
                THCudaTensor_select(state, single_gradInput, gradInput, 0, i);
                THCudaTensor_select(state, single_gradOutput, gradOutput, 0, i);
                
                cAngles2BasisDihedral a2b(state, single_B, single_input_angles, single_daxis_dangle, R, THIntTensor_get1d(angles_length, i));
                a2b.computeBackward(single_gradInput, single_gradOutput);

                THCudaTensor_free(state, single_input_angles);
                THCudaTensor_free(state, single_B);
                THCudaTensor_free(state, single_daxis_dangle);
                THCudaTensor_free(state, single_gradInput);
                THCudaTensor_free(state, single_gradOutput);
            }
        }
    }


}