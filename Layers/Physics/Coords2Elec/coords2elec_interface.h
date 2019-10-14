#include <torch/extension.h>
void Coords2Eps_forward(torch::Tensor coords, torch::Tensor assigned_params, torch::Tensor num_atoms, torch::Tensor eps, 
                        float resolution, float stern_size);
void Coords2Eps_backward(   torch::Tensor gradOutput, torch::Tensor gradInput, 
                            torch::Tensor coords, torch::Tensor assigned_params, torch::Tensor num_atoms,
                            float resolution);

void Coords2Q_forward(torch::Tensor coords, torch::Tensor assigned_params, torch::Tensor num_atoms, torch::Tensor Q, float resolution);

void QEps2Phi_forward(  torch::Tensor Q, torch::Tensor Eps, torch::Tensor Phi, 
                        float resolution, float kappa02);