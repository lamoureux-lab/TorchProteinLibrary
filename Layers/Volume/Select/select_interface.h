#include <torch/extension.h>

void SelectVolume_forward(  torch::Tensor volume,
                            torch::Tensor coords,
                            torch::Tensor num_atoms,
                            torch::Tensor features,
                            float res
                        );