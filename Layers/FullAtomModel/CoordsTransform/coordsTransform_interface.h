#include <torch/extension.h>
void CoordsTranslate_forward(   torch::Tensor input_coords, 
                                torch::Tensor output_coords,
                                torch::Tensor T,
                                torch::Tensor num_atoms
                                );

void CoordsTranslate_backward(  torch::Tensor grad_output_coords, 
                                torch::Tensor grad_input_coords,
                                torch::Tensor T,
                                torch::Tensor num_atoms
                                );

void CoordsRotate_forward(  torch::Tensor input_coords, 
                            torch::Tensor output_coords,
                            torch::Tensor R,
                            torch::Tensor num_atoms
                            );

void CoordsRotate_backward( torch::Tensor grad_output_coords, 
                            torch::Tensor grad_input_coords,
                            torch::Tensor R,
                            torch::Tensor num_atoms);


void Coords2Center_forward(     torch::Tensor input_coords, 
                                torch::Tensor output_T,
                                torch::Tensor num_atoms);

void Coords2Center_backward(    torch::Tensor grad_output_T, 
                                torch::Tensor grad_input_coords,
                                torch::Tensor num_atoms);

void getBBox(   torch::Tensor input_coords,
                torch::Tensor a, torch::Tensor b,
                torch::Tensor num_atoms);

void getRandomRotation( torch::Tensor R);
void getRotation( torch::Tensor R, torch::Tensor u );
void getRandomTranslation( torch::Tensor T, torch::Tensor a, torch::Tensor b, float volume_size);
