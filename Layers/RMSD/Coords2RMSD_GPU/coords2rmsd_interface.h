#include <torch/torch.h>
void Coords2RMSD_GPU_forward(   at::Tensor re_coordinates_src, at::Tensor re_coordinates_dst, 
                                at::Tensor output, at::Tensor num_atoms,
                                at::Tensor Ut_coordinates_dst
                            );