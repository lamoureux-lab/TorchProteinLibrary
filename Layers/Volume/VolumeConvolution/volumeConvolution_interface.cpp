#include "volumeConvolution_interface.h"
#include <VolumeConv.h>
#include <iostream>
#include <nUtil.h>


void VolumeConvolution_forward( torch::Tensor volume1, 
                                torch::Tensor volume2, 
                                torch::Tensor output){
    CHECK_GPU_INPUT_TYPE(volume1, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(volume2, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(output, torch::kFloat);
    if(volume1.ndimension()!=4){
        ERROR("incorrect input dimension");
    }
    cpu_VolumeConv(	volume1.data<float>(), 
                    volume2.data<float>(), 
                    output.data<float>(), 
                    volume1.size(0),
                    volume1.size(1),
                    true);
}
void VolumeConvolution_backward(    torch::Tensor gradOutput,
                                    torch::Tensor gradVolume1,
                                    torch::Tensor gradVolume2,
                                    torch::Tensor volume1, 
                                    torch::Tensor volume2){
    CHECK_GPU_INPUT_TYPE(gradVolume1, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(gradVolume2, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(gradOutput, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(volume1, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(volume2, torch::kFloat);
    if(gradOutput.ndimension()!=4){
        ERROR("incorrect input dimension");
    }

    cpu_VolumeConv(	gradOutput.data<float>(), 
                    volume2.data<float>(), 
                    gradVolume1.data<float>(), 
                    volume1.size(0),
                    volume1.size(1),
                    false);

    cpu_VolumeConv(	volume1.data<float>(), 
                    gradOutput.data<float>(),
                    gradVolume2.data<float>(), 
                    volume1.size(0),
                    volume1.size(1),
                    true);    
}


