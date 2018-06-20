#include <THC/THC.h>
#include <cTensorProteinCUDAKernels.h>
#include <iostream>

extern THCState *state;
#define CUDA_REAL_TENSOR_VAR THCudaTensor
#define CUDA_REAL_TENSOR(X) THCudaTensor_##X

extern "C" {
    int Angles2Coords_forward(  CUDA_REAL_TENSOR_VAR *input_angles, 
                                CUDA_REAL_TENSOR_VAR *output_coords, 
                                THCudaIntTensor *angles_length, 
                                CUDA_REAL_TENSOR_VAR *A
                            ){
        if(input_angles->nDimension!=3){
            std::cout<<"incorrect input dimension"<<input_angles->nDimension<<std::endl;
            throw("incorrect input dimension");
        }
        cpu_computeCoordinatesDihedral(	CUDA_REAL_TENSOR(data)(state, input_angles), 
							CUDA_REAL_TENSOR(data)(state, output_coords), 
							CUDA_REAL_TENSOR(data)(state, A), 
							THCudaIntTensor_data(state, angles_length),
                            input_angles->size[0],
                            input_angles->size[2]);
    }
    int Angles2Coords_backward(     CUDA_REAL_TENSOR_VAR *gradInput,
                                    CUDA_REAL_TENSOR_VAR *gradOutput,
                                    CUDA_REAL_TENSOR_VAR *input_angles, 
                                    THCudaIntTensor *angles_length, 
                                    CUDA_REAL_TENSOR_VAR *A,   
                                    CUDA_REAL_TENSOR_VAR *dr_dangle
                            ){
        if(gradInput->nDimension!=3){
            std::cout<<"incorrect input dimension"<<gradInput->nDimension<<std::endl;
            throw("incorrect input dimension");
        }
        cpu_computeDerivativesDihedral( CUDA_REAL_TENSOR(data)(state, input_angles),
                                        CUDA_REAL_TENSOR(data)(state, dr_dangle),
                                        CUDA_REAL_TENSOR(data)(state, A),
                                        THCudaIntTensor_data(state, angles_length),
                                        input_angles->size[0],
                                        input_angles->size[2]);
        
        cpu_backwardFromCoords( CUDA_REAL_TENSOR(data)(state, gradInput),
                                CUDA_REAL_TENSOR(data)(state, gradOutput),
                                CUDA_REAL_TENSOR(data)(state, dr_dangle),
                                THCudaIntTensor_data(state, angles_length),
                                input_angles->size[0],
                                input_angles->size[2]);
    }


}