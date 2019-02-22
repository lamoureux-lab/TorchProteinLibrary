#include <torch/torch.h>
void TypedCoords2Volume_forward(    at::Tensor input_coords,
                                        at::Tensor volume,
                                        at::Tensor num_atoms_of_type,
                                        at::Tensor offsets, 
                                        float resolution, int mode);
void TypedCoords2Volume_backward(   at::Tensor grad_volume,
                                        at::Tensor grad_coords,
                                        at::Tensor coords,
                                        at::Tensor num_atoms_of_type,
                                        at::Tensor offsets, 
                                        float resolution, int mode);