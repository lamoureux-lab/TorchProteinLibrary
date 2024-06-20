#include <torch/extension.h>
void Coords2Stress_forward( torch::Tensor coords, torch::Tensor vectors, torch::Tensor num_atoms, torch::Tensor volume, 
                            float resolution);