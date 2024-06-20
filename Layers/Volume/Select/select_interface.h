#include <torch/extension.h>

void SelectVolume_forward(  torch::Tensor volume,
                            torch::Tensor coords,
                            torch::Tensor num_atoms,
                            torch::Tensor features,
                            float res
                        );

void SelectVolume_backward( torch::Tensor gradOutput,
                            torch::Tensor gradInput,
                            torch::Tensor coords,
                            torch::Tensor num_atoms,
                            float res
                        );