#include <torch/extension.h>
void VolumeGenRMSD( torch::Tensor coords,
                    torch::Tensor num_atoms,
                    torch::Tensor R0,
                    torch::Tensor R1,
                    torch::Tensor T0,
                    torch::Tensor translation_volume,
                    float resolution);