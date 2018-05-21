#include <THC/THC.h>
#include <cTensorProteinCUDAKernels.h>
#include <iostream>

extern THCState *state;
float R = 3.8;
extern "C" {
    int Angles2Coords_forward(  THCudaDoubleTensor *input_angles, 
                                THCudaDoubleTensor *output_coords, 
                                THCudaIntTensor *angles_length, 
                                THCudaDoubleTensor *A
                            ){

        cpu_computeCoordinatesDihedral(	THCudaDoubleTensor_data(state, input_angles), 
							THCudaDoubleTensor_data(state, output_coords), 
							THCudaDoubleTensor_data(state, A), 
							THCudaIntTensor_data(state, angles_length),
                            input_angles->size[0],
                            input_angles->size[2]);
    }
    int Angles2Coords_backward(     THCudaDoubleTensor *gradInput,
                                    THCudaDoubleTensor *gradOutput,
                                    THCudaDoubleTensor *input_angles, 
                                    THCudaIntTensor *angles_length, 
                                    THCudaDoubleTensor *A,   
                                    THCudaDoubleTensor *dr_dangle
                            ){
        // #pragma omp parallel for num_threads(10)
        // for(int i=0; i<input_angles->size[0]; i++){
        //     THCudaTensor *single_input_angles, *single_A, *single_dr_dangle, *single_gradInput, *single_gradOutput;
        //     single_input_angles = THCudaTensor_new(state);
        //     single_A = THCudaTensor_new(state);
        //     single_dr_dangle = THCudaTensor_new(state);
        //     single_gradInput = THCudaTensor_new(state);
        //     single_gradOutput = THCudaTensor_new(state);

        //     THCudaTensor_select(state, single_input_angles, input_angles, 0, i);
        //     THCudaTensor_select(state, single_A, A, 0, i);
        //     THCudaTensor_select(state, single_dr_dangle, dr_dangle, 0, i);
        //     THCudaTensor_select(state, single_gradInput, gradInput, 0, i);
        //     THCudaTensor_select(state, single_gradOutput, gradOutput, 0, i);
            
        //     cAngles2CoordsDihedral a2c(state, single_A, single_input_angles, single_dr_dangle, R, THIntTensor_get1d(angles_length, i));
        //     a2c.computeBackward(single_gradInput, single_gradOutput);

        //     THCudaTensor_free(state, single_input_angles);
        //     THCudaTensor_free(state, single_A);
        //     THCudaTensor_free(state, single_dr_dangle);
        //     THCudaTensor_free(state, single_gradInput);
        //     THCudaTensor_free(state, single_gradOutput);
        // }
        
    }


}