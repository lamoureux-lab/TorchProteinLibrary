#include <torch/extension.h>
void Coords2RMSD_CPU_forward(   torch::Tensor src, torch::Tensor dst, torch::Tensor rmsd,
                            torch::Tensor ce_src, torch::Tensor ce_dst,
                            torch::Tensor U_ce_src, torch::Tensor UT_ce_dst,
                            torch::Tensor num_atoms
                        );
void Coords2RMSD_CPU_backward(  torch::Tensor grad_atoms, torch::Tensor grad_output,
                            torch::Tensor ce_src, torch::Tensor ce_dst,
                            torch::Tensor U_ce_src, torch::Tensor UT_ce_dst,
                            torch::Tensor num_atoms
                        );