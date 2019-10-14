#include <torch/extension.h>
void Coords2Stress_forward( torch::Tensor coords, torch::Tensor gamma, torch::Tensor num_atoms, torch::Tensor stress, 
                            float resolution);