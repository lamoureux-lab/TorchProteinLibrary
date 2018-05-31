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
        if(input_angles->nDimension!=3){
            std::cout<<"incorrect input dimension"<<input_angles->nDimension<<std::endl;
            throw("incorrect input dimension");
        }
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
        if(gradInput->nDimension!=3){
            std::cout<<"incorrect input dimension"<<gradInput->nDimension<<std::endl;
            throw("incorrect input dimension");
        }
        cpu_computeDerivativesDihedral( THCudaDoubleTensor_data(state, input_angles),
                                        THCudaDoubleTensor_data(state, dr_dangle),
                                        THCudaDoubleTensor_data(state, A),
                                        THCudaIntTensor_data(state, angles_length),
                                        input_angles->size[0],
                                        input_angles->size[2]);
        
        cpu_backwardFromCoords( THCudaDoubleTensor_data(state, gradInput),
                                THCudaDoubleTensor_data(state, gradOutput),
                                THCudaDoubleTensor_data(state, dr_dangle),
                                THCudaIntTensor_data(state, angles_length),
                                input_angles->size[0],
                                input_angles->size[2]);
    }


}