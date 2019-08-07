#include "angles2backbone_interface.h"
#include <cBackboneProteinCUDAKernels.h>
#include <cBackboneProteinCPUKernels.hpp>
#include <iostream>
#include <nUtil.h>

int Angles2BackboneGPU_forward(torch::Tensor input_angles, 
                            torch::Tensor output_coords, 
                            torch::Tensor angles_length, 
                            torch::Tensor A
                        ){
    CHECK_GPU_INPUT(input_angles);
    CHECK_GPU_INPUT(output_coords);
    CHECK_GPU_INPUT(A);
    CHECK_GPU_INPUT_TYPE(angles_length, torch::kInt);
    if(input_angles.ndimension() != 3){
        ERROR("Incorrect input ndim");
    }
    
    gpu_computeCoordinatesBackbone(	input_angles.data<float>(), 
                                    output_coords.data<float>(), 
                                    A.data<float>(), 
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
}
int Angles2BackboneGPU_backward(   torch::Tensor gradInput,
                                torch::Tensor gradOutput,
                                torch::Tensor input_angles, 
                                torch::Tensor angles_length, 
                                torch::Tensor A,   
                                torch::Tensor dr_dangle
                            ){
    CHECK_GPU_INPUT(gradInput);
    CHECK_GPU_INPUT(gradOutput);
    CHECK_GPU_INPUT(input_angles);
    CHECK_GPU_INPUT(dr_dangle);
    CHECK_GPU_INPUT(A);
    CHECK_GPU_INPUT_TYPE(angles_length, torch::kInt);
    if(gradOutput.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    gpu_computeDerivativesBackbone( input_angles.data<float>(),
                                    dr_dangle.data<float>(),
                                    A.data<float>(),
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
    
    gpu_backwardFromCoordsBackbone( gradInput.data<float>(),
                                    gradOutput.data<float>(),
                                    dr_dangle.data<float>(),
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
}



int Angles2BackboneCPU_forward(torch::Tensor input_angles, 
                            torch::Tensor output_coords, 
                            torch::Tensor angles_length, 
                            torch::Tensor A
                        ){
    CHECK_CPU_INPUT(input_angles);
    CHECK_CPU_INPUT(output_coords);
    CHECK_CPU_INPUT(A);
    CHECK_CPU_INPUT_TYPE(angles_length, torch::kInt);
    if(input_angles.ndimension() != 3){
        ERROR("Incorrect input ndim");
    }
    
    cpu_computeCoordinatesBackbone<double>(	input_angles.data<double>(), 
                                    output_coords.data<double>(), 
                                    A.data<double>(), 
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
}


int Angles2BackboneCPU_backward(torch::Tensor gradInput,
                                torch::Tensor gradOutput,
                                torch::Tensor input_angles, 
                                torch::Tensor angles_length, 
                                torch::Tensor A,   
                                torch::Tensor dr_dangle
                            ){
    CHECK_CPU_INPUT(gradInput);
    CHECK_CPU_INPUT(gradOutput);
    CHECK_CPU_INPUT(input_angles);
    CHECK_CPU_INPUT(dr_dangle);
    CHECK_CPU_INPUT(A);
    CHECK_CPU_INPUT_TYPE(angles_length, torch::kInt);
    if(gradOutput.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    
    cpu_computeDerivativesBackbone<double>( input_angles.data<double>(),
                                    dr_dangle.data<double>(),
                                    A.data<double>(),
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
    
    cpu_backwardFromCoordsBackbone<double>( gradInput.data<double>(),
                                    gradOutput.data<double>(),
                                    dr_dangle.data<double>(),
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
}