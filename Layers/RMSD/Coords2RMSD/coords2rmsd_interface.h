#include <torch/extension.h>
void Coords2RMSDGPU_forward(   torch::Tensor centered_coords_src, 
                                torch::Tensor centered_coords_dst, 
                                torch::Tensor output, 
                                torch::Tensor num_atoms,
                                torch::Tensor UT
                        );
void Coords2RMSD_forward(   torch::Tensor centered_coords_src, 
                            torch::Tensor centered_coords_dst, 
                            torch::Tensor output, 
                            torch::Tensor num_atoms,
                            torch::Tensor UT
                        );