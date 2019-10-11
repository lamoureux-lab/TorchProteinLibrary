#include <torch/extension.h>
void Coords2Eps_forward(torch::Tensor coords, torch::Tensor assigned_params, torch::Tensor num_atoms, torch::Tensor eps, float resolution);
void Coords2Eps_backward(   torch::Tensor gradOutput, torch::Tensor gradInput, 
                            torch::Tensor coords, torch::Tensor assigned_params, torch::Tensor num_atoms,
                            float resolution);