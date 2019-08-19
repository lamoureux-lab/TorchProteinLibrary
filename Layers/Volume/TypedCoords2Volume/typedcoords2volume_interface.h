#include <torch/extension.h>
void TypedCoords2Volume_forward(    torch::Tensor input_coords,
                                        torch::Tensor volume,
                                        torch::Tensor num_atoms_of_type,
                                        torch::Tensor offsets, 
                                        float resolution);
void TypedCoords2Volume_backward(   torch::Tensor grad_volume,
                                        torch::Tensor grad_coords,
                                        torch::Tensor coords,
                                        torch::Tensor num_atoms_of_type,
                                        torch::Tensor offsets, 
                                        float resolution);