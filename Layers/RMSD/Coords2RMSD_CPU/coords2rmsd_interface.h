#include <torch/torch.h>
void Coords2RMSD_CPU_forward(   at::Tensor src, at::Tensor dst, at::Tensor rmsd,
                            at::Tensor ce_src, at::Tensor ce_dst,
                            at::Tensor U_ce_src, at::Tensor UT_ce_dst,
                            at::Tensor num_atoms
                        );
void Coords2RMSD_CPU_backward(  at::Tensor grad_atoms, at::Tensor grad_output,
                            at::Tensor ce_src, at::Tensor ce_dst,
                            at::Tensor U_ce_src, at::Tensor UT_ce_dst,
                            at::Tensor num_atoms
                        );