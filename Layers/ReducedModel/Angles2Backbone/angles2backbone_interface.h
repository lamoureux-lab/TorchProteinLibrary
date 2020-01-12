#include <torch/extension.h>

int Angles2BackboneGPU_forward(torch::Tensor input_angles, 
                            torch::Tensor param,
                            torch::Tensor output_coords, 
                            torch::Tensor angles_length, 
                            torch::Tensor A
                            );
int Angles2BackboneGPUAngles_backward(  torch::Tensor gradInput,
                                        torch::Tensor gradOutput,
                                        torch::Tensor input_angles, 
                                        torch::Tensor param, 
                                        torch::Tensor angles_length, 
                                        torch::Tensor A,   
                                        torch::Tensor dr_dangle
                            );
int Angles2BackboneGPUParam_backward(torch::Tensor gradParam,
                                torch::Tensor gradOutput,
                                torch::Tensor input_angles, 
                                torch::Tensor param, 
                                torch::Tensor angles_length, 
                                torch::Tensor A,
                                torch::Tensor dr_dparam
                            );

int Angles2BackboneCPU_forward(torch::Tensor input_angles, 
                                torch::Tensor param,
                                torch::Tensor output_coords, 
                                torch::Tensor angles_length, 
                                torch::Tensor A
                            );

int Angles2BackboneCPUAngles_backward(torch::Tensor gradInput,
                                torch::Tensor gradOutput,
                                torch::Tensor input_angles, 
                                torch::Tensor param,
                                torch::Tensor angles_length, 
                                torch::Tensor A,   
                                torch::Tensor dr_dangle
                            );
int Angles2BackboneCPUParam_backward(torch::Tensor gradParam,
                                torch::Tensor gradOutput,
                                torch::Tensor input_angles, 
                                torch::Tensor param,
                                torch::Tensor angles_length, 
                                torch::Tensor A,
                                torch::Tensor dr_dparam
                            );