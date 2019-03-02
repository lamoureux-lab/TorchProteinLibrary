#include <torch/torch.h>

int Angles2BackboneGPU_forward(at::Tensor input_angles, 
                            at::Tensor output_coords, 
                            at::Tensor angles_length, 
                            at::Tensor A
                            );

int Angles2BackboneGPU_backward(   at::Tensor gradInput,
                                at::Tensor gradOutput,
                                at::Tensor input_angles, 
                                at::Tensor angles_length, 
                                at::Tensor A,   
                                at::Tensor dr_dangle
                            );

int Angles2BackboneCPU_forward(at::Tensor input_angles, 
                            at::Tensor output_coords, 
                            at::Tensor angles_length, 
                            at::Tensor A
                            );

int Angles2BackboneCPU_backward(   at::Tensor gradInput,
                                at::Tensor gradOutput,
                                at::Tensor input_angles, 
                                at::Tensor angles_length, 
                                at::Tensor A,   
                                at::Tensor dr_dangle
                            );