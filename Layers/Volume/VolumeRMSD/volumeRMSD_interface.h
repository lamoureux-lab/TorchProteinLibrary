#include <torch/torch.h>
void VolumeGenRMSD( at::Tensor coords,
                    at::Tensor num_atoms,
                    at::Tensor R0,
                    at::Tensor R1,
                    at::Tensor T0,
                    at::Tensor translation_volume,
                    float resolution);