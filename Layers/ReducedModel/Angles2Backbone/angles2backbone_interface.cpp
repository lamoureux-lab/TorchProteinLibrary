#include <THC/THC.h>
#include <cBackboneProteinCUDAKernels.h>
#include <iostream>

extern THCState *state;
#define CUDA_REAL_TENSOR_VAR THCudaTensor
#define CUDA_REAL_TENSOR(X) THCudaTensor_##X
// #define CUDA_REAL_TENSOR_VAR THCudaDoubleTensor
// #define CUDA_REAL_TENSOR(X) THCudaDoubleTensor_##X
bool int2bool(int add_terminal){
    bool add_term;
    if(add_terminal == 1){
        add_term = true;
    }else if(add_terminal == 0){
        add_term = false;
    }else{
        std::cout<<"unknown add_terminal = "<<add_terminal<<std::endl;
        throw std::string("unknown add_terminal");
    }
    return add_term;
}
extern "C" {
    int Angles2Backbone_forward(  CUDA_REAL_TENSOR_VAR *input_angles, 
                                CUDA_REAL_TENSOR_VAR *output_coords, 
                                THCudaIntTensor *angles_length, 
                                CUDA_REAL_TENSOR_VAR *A
                            ){
        if(input_angles->nDimension!=3){
            std::cout<<"incorrect input dimension"<<input_angles->nDimension<<std::endl;
            throw("incorrect input dimension");
        }
        cpu_computeCoordinatesBackbone(	CUDA_REAL_TENSOR(data)(state, input_angles), 
							CUDA_REAL_TENSOR(data)(state, output_coords), 
							CUDA_REAL_TENSOR(data)(state, A), 
							THCudaIntTensor_data(state, angles_length),
                            input_angles->size[0],
                            input_angles->size[2]);
    }
    int Angles2Backbone_backward(   CUDA_REAL_TENSOR_VAR *gradInput,
                                    CUDA_REAL_TENSOR_VAR *gradOutput,
                                    CUDA_REAL_TENSOR_VAR *input_angles, 
                                    THCudaIntTensor *angles_length, 
                                    CUDA_REAL_TENSOR_VAR *A,   
                                    CUDA_REAL_TENSOR_VAR *dr_dangle,
                                    int norm
                            ){
        if(gradInput->nDimension!=3){
            std::cout<<"incorrect input dimension"<<gradInput->nDimension<<std::endl;
            throw("incorrect input dimension");
        }
        bool bnorm = int2bool(norm);
        cpu_computeDerivativesBackbone( CUDA_REAL_TENSOR(data)(state, input_angles),
                                        CUDA_REAL_TENSOR(data)(state, dr_dangle),
                                        CUDA_REAL_TENSOR(data)(state, A),
                                        THCudaIntTensor_data(state, angles_length),
                                        input_angles->size[0],
                                        input_angles->size[2]);
        
        cpu_backwardFromCoordsBackbone( CUDA_REAL_TENSOR(data)(state, gradInput),
                                CUDA_REAL_TENSOR(data)(state, gradOutput),
                                CUDA_REAL_TENSOR(data)(state, dr_dangle),
                                THCudaIntTensor_data(state, angles_length),
                                input_angles->size[0],
                                input_angles->size[2],
                                bnorm);
    }


}