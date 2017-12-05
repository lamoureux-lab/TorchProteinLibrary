#include <THC/THC.h>
#include "cDdist2Forces.h"
#include <iostream>

extern THCState *state;
extern "C" {
    int Ddist2Forces_forward(   THCudaTensor *input_ddist, 
                                THCudaTensor *input_coords, 
                                THCudaTensor *output_forces, 
                                THIntTensor *angles_length
                            ){
        if(input_ddist->nDimension == 2){
            cDdist2Forces dd2f(state, input_coords, THIntTensor_get1d(angles_length, 0));
            dd2f.computeForward(input_ddist, output_forces);
        }else{
            // #pragma omp parallel for num_threads(10)
            for(int i=0; i<input_ddist->size[0]; i++){
                THCudaTensor *single_input_ddist, *single_input_coords, *single_output_forces;
                single_input_ddist = THCudaTensor_new(state);
	            single_input_coords = THCudaTensor_new(state);
                single_output_forces = THCudaTensor_new(state);

                THCudaTensor_select(state, single_input_ddist, input_ddist, 0, i);
                THCudaTensor_select(state, single_input_coords, input_coords, 0, i);
                THCudaTensor_select(state, single_output_forces, output_forces, 0, i);
                cDdist2Forces dd2f(state, single_input_coords, THIntTensor_get1d(angles_length, i));
                dd2f.computeForward(single_input_ddist, single_output_forces);

                THCudaTensor_free(state, single_input_ddist);
                THCudaTensor_free(state, single_input_coords);
                THCudaTensor_free(state, single_output_forces);
            }
        }
    }
    // int Angles2Coords_backward(     THCudaTensor *gradInput,
    //                                 THCudaTensor *gradOutput,
    //                                 THCudaTensor *input_angles, 
    //                                 THIntTensor *angles_length, 
    //                                 THCudaTensor *A,   
    //                                 THCudaTensor *dr_dangle
    //                         ){
    //     if(input_angles->nDimension == 2){
    //         cAngles2CoordsAB a2c(state, A, input_angles, dr_dangle, THIntTensor_get1d(angles_length, 0));
    //         a2c.computeBackward(gradInput, gradOutput);
    //     }else{
    //         #pragma omp parallel for num_threads(10)
    //         for(int i=0; i<input_angles->size[0]; i++){
    //             THCudaTensor *single_input_angles, *single_A, *single_dr_dangle, *single_gradInput, *single_gradOutput;
    //             single_input_angles = THCudaTensor_new(state);
	//             single_A = THCudaTensor_new(state);
    //             single_dr_dangle = THCudaTensor_new(state);
    //             single_gradInput = THCudaTensor_new(state);
    //             single_gradOutput = THCudaTensor_new(state);

    //             THCudaTensor_select(state, single_input_angles, input_angles, 0, i);
    //             THCudaTensor_select(state, single_A, A, 0, i);
    //             THCudaTensor_select(state, single_dr_dangle, dr_dangle, 0, i);
    //             THCudaTensor_select(state, single_gradInput, gradInput, 0, i);
    //             THCudaTensor_select(state, single_gradOutput, gradOutput, 0, i);
                
    //             cAngles2CoordsAB a2c(state, single_A, single_input_angles, single_dr_dangle, THIntTensor_get1d(angles_length, i));
    //             a2c.computeBackward(single_gradInput, single_gradOutput);

    //             THCudaTensor_free(state, single_input_angles);
    //             THCudaTensor_free(state, single_A);
    //             THCudaTensor_free(state, single_dr_dangle);
    //             THCudaTensor_free(state, single_gradInput);
    //             THCudaTensor_free(state, single_gradOutput);
    //         }
    //     }
    // }


}