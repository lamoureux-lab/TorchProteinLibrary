#include <torch/extension.h>

int Angles2BackboneGPU_forward(torch::Tensor input_angles, 
                            torch::Tensor output_coords, 
                            torch::Tensor angles_length, 
                            torch::Tensor A
                            );

int Angles2BackboneGPU_backward(   torch::Tensor gradInput,
                                torch::Tensor gradOutput,
                                torch::Tensor input_angles, 
                                torch::Tensor angles_length, 
                                torch::Tensor A,   
                                torch::Tensor dr_dangle
                            );

int Angles2BackboneCPU_forward(torch::Tensor input_angles, 
                            torch::Tensor output_coords, 
                            torch::Tensor angles_length, 
                            torch::Tensor A
                            );

int Angles2BackboneCPU_backward(   torch::Tensor gradInput,
                                torch::Tensor gradOutput,
                                torch::Tensor input_angles, 
                                torch::Tensor angles_length, 
                                torch::Tensor A,   
                                torch::Tensor dr_dangle
                            );