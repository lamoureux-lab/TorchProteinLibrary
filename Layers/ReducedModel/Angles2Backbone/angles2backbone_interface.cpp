#include "angles2backbone_interface.h"
#include <cBackboneProteinCUDAKernels.h>
#include <cBackboneProteinCPUKernels.hpp>
#include <iostream>
#include <nUtil.h>

int Angles2BackboneGPU_forward(torch::Tensor input_angles, 
                            torch::Tensor param, 
                            torch::Tensor output_coords, 
                            torch::Tensor angles_length, 
                            torch::Tensor A
                        ){
    CHECK_GPU_INPUT(input_angles);
    CHECK_GPU_INPUT(param);
    CHECK_GPU_INPUT(output_coords);
    CHECK_GPU_INPUT(A);
    CHECK_GPU_INPUT_TYPE(angles_length, torch::kInt);
    if(input_angles.ndimension() != 3){
        ERROR("Incorrect input ndim");
    }
    
    gpu_computeCoordinatesBackbone<float>(	input_angles.data<float>(), 
                                    param.data<float>(), 
                                    output_coords.data<float>(), 
                                    A.data<float>(), 
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
}
int Angles2BackboneGPU_backward(   torch::Tensor gradInput,
                                torch::Tensor gradParam,
                                torch::Tensor gradOutput,
                                torch::Tensor input_angles, 
                                torch::Tensor param, 
                                torch::Tensor angles_length, 
                                torch::Tensor A,   
                                torch::Tensor dr_dangle,
                                torch::Tensor dr_dparam
                            ){
    CHECK_GPU_INPUT(gradInput);
    CHECK_GPU_INPUT(gradParam);
    CHECK_GPU_INPUT(gradOutput);
    CHECK_GPU_INPUT(input_angles);
    CHECK_GPU_INPUT(param);
    CHECK_GPU_INPUT(dr_dangle);
    CHECK_GPU_INPUT(dr_dparam);
    CHECK_GPU_INPUT(A);
    CHECK_GPU_INPUT_TYPE(angles_length, torch::kInt);
    if(gradOutput.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    gpu_computeDerivativesBackbone<float>( input_angles.data<float>(),
                                    param.data<float>(),
                                    dr_dangle.data<float>(),
                                    A.data<float>(),
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
    
    gpu_backwardFromCoordsBackbone<float>( gradInput.data<float>(),
                                    gradOutput.data<float>(),
                                    dr_dangle.data<float>(),
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
    
    gpu_computeDerivativesParam<float>( input_angles.data<float>(),
                                    param.data<float>(),
                                    dr_dparam.data<float>(),
                                    A.data<float>(),
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
    
    gpu_backwardFromCoordsParam<float>( gradParam.data<float>(),
                                    gradOutput.data<float>(),
                                    dr_dangle.data<float>(),
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
}



int Angles2BackboneCPU_forward( torch::Tensor input_angles, 
                                torch::Tensor param,
                                torch::Tensor output_coords, 
                                torch::Tensor angles_length, 
                                torch::Tensor A
                            ){
    CHECK_CPU_INPUT(input_angles);
    CHECK_CPU_INPUT(param);
    CHECK_CPU_INPUT(output_coords);
    CHECK_CPU_INPUT(A);
    CHECK_CPU_INPUT_TYPE(angles_length, torch::kInt);
    if(input_angles.ndimension() != 3){
        ERROR("Incorrect input ndim");
    }
    
    cpu_computeCoordinatesBackbone<double>(	input_angles.data<double>(), 
                                            param.data<double>(),
                                            output_coords.data<double>(), 
                                            A.data<double>(), 
                                            angles_length.data<int>(),
                                            input_angles.size(0),
                                            input_angles.size(2));
}


int Angles2BackboneCPU_backward(torch::Tensor gradInput,
                                torch::Tensor gradParam,
                                torch::Tensor gradOutput,
                                torch::Tensor input_angles, 
                                torch::Tensor param,
                                torch::Tensor angles_length, 
                                torch::Tensor A,   
                                torch::Tensor dr_dangle,
                                torch::Tensor dr_dparam
                            ){
    CHECK_CPU_INPUT(gradInput);
    CHECK_CPU_INPUT(gradParam);
    CHECK_CPU_INPUT(gradOutput);
    CHECK_CPU_INPUT(input_angles);
    CHECK_CPU_INPUT(param);
    CHECK_CPU_INPUT(dr_dangle);
    CHECK_CPU_INPUT(dr_dparam);
    CHECK_CPU_INPUT(A);
    CHECK_CPU_INPUT_TYPE(angles_length, torch::kInt);
    if(gradOutput.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    
    cpu_computeDerivativesBackbone<double>( input_angles.data<double>(),
                                    param.data<double>(),
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
    // std::cout<<"deriv"<<std::endl;
    cpu_computeDerivativesParam<double>( input_angles.data<double>(),
                                        param.data<double>(),
                                        dr_dparam.data<double>(),
                                        A.data<double>(),
                                        angles_length.data<int>(),
                                        input_angles.size(0),
                                        input_angles.size(2));
    // std::cout<<"reduce"<<std::endl;
    cpu_backwardFromCoordsParam<double>( gradParam.data<double>(),
                                            gradOutput.data<double>(),
                                            dr_dparam.data<double>(),
                                            angles_length.data<int>(),
                                            input_angles.size(0),
                                            input_angles.size(2));
    // std::cout<<"end"<<std::endl;
}