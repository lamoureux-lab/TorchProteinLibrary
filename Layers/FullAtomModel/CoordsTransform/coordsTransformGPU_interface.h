#include <torch/extension.h>
void CoordsTranslateGPU_forward(    torch::Tensor input_coords, 
                                    torch::Tensor output_coords,
                                    torch::Tensor T,
                                    torch::Tensor num_atoms
                                );

void CoordsTranslateGPU_backward(   torch::Tensor grad_output_coords, 
                                    torch::Tensor grad_input_coords,
                                    torch::Tensor T,
                                    torch::Tensor num_atoms
                                );

void CoordsRotateGPU_forward(   torch::Tensor input_coords, 
                                torch::Tensor output_coords,
                                torch::Tensor R,
                                torch::Tensor num_atoms
                            );

void CoordsRotateGPU_backward(  torch::Tensor grad_output_coords, 
                                torch::Tensor grad_input_coords,
                                torch::Tensor R,
                                torch::Tensor num_atoms);