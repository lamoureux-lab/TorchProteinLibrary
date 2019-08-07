#include <torch/extension.h>
void Coords2RMSD_GPU_forward(   torch::Tensor re_coordinates_src, torch::Tensor re_coordinates_dst, 
                                torch::Tensor output, torch::Tensor num_atoms,
                                torch::Tensor Ut_coordinates_dst
                            );