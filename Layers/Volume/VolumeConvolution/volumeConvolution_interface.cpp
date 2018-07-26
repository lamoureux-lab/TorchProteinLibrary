#include <THC/THC.h>
#include <VolumeConv.h>
#include <iostream>

extern THCState *state;
#define CUDA_REAL_TENSOR_VAR THCudaTensor
#define CUDA_REAL_TENSOR(X) THCudaTensor_##X
// #define CUDA_REAL_TENSOR_VAR THCudaDoubleTensor
// #define CUDA_REAL_TENSOR(X) THCudaDoubleTensor_##X

extern "C" {
    void VolumeConvolution_forward(  CUDA_REAL_TENSOR_VAR *volume1, 
                                    CUDA_REAL_TENSOR_VAR *volume2, 
                                    CUDA_REAL_TENSOR_VAR *output){
        if(volume1->nDimension!=4){
            std::cout<<"incorrect input dimension"<<volume1->nDimension<<std::endl;
            throw("incorrect input dimension");
        }
        cpu_VolumeConv(	CUDA_REAL_TENSOR(data)(state, volume1), 
							CUDA_REAL_TENSOR(data)(state, volume2), 
							CUDA_REAL_TENSOR(data)(state, output), 
							volume1->size[0],
                            volume1->size[1]);
    }
    void VolumeConvolution_backward( CUDA_REAL_TENSOR_VAR *gradOutput,
                                    CUDA_REAL_TENSOR_VAR *gradVolume1,
                                    CUDA_REAL_TENSOR_VAR *gradVolume2,
                                    CUDA_REAL_TENSOR_VAR *volume1, 
                                    CUDA_REAL_TENSOR_VAR *volume2){
        if(gradOutput->nDimension!=4){
            std::cout<<"incorrect input dimension"<<gradOutput->nDimension<<std::endl;
            throw("incorrect input dimension");
        }
        
        cpu_VolumeConvGrad( CUDA_REAL_TENSOR(data)(state, gradOutput),
                            CUDA_REAL_TENSOR(data)(state, volume1),
                            CUDA_REAL_TENSOR(data)(state, volume2),
                            CUDA_REAL_TENSOR(data)(state, gradVolume1),
                            volume1->size[0], volume1->size[1], true);
        
        cpu_VolumeConvGrad( CUDA_REAL_TENSOR(data)(state, gradOutput),
                            CUDA_REAL_TENSOR(data)(state, volume2),
                            CUDA_REAL_TENSOR(data)(state, volume1),
                            CUDA_REAL_TENSOR(data)(state, gradVolume2),
                            volume1->size[0], volume1->size[1], false);
    }


}