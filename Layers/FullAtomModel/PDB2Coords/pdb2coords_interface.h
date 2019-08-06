#include <torch/extension.h>
// void PDB2CoordsOrdered(at::Tensor filenames, at::Tensor coords, at::Tensor res_names, at::Tensor atom_names, at::Tensor mask);
void PDB2CoordsUnordered(at::Tensor filenames, at::Tensor coords, at::Tensor chain_names, at::Tensor res_names, at::Tensor res_nums, at::Tensor atom_names, at::Tensor num_atoms);
