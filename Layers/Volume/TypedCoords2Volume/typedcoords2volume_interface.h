#include <torch/extension.h>
void TypedCoords2Volume_forward(    torch::Tensor input_coords,
                                    torch::Tensor volume,
                                    torch::Tensor num_atoms,
                                    float resolution, int num_neighbours,
                                    torch::Tensor gridParticleHash,
                                    torch::Tensor gridParticleIndex,
                                    torch::Tensor cellStart,
                                    torch::Tensor cellStop,
                                    torch::Tensor sortedPos);

void TypedCoords2Volume_backward(   torch::Tensor grad_volume,
                                    torch::Tensor grad_coords,
                                    torch::Tensor coords,
                                    torch::Tensor num_atoms,
                                    float resolution,
                                    int num_neighbours);