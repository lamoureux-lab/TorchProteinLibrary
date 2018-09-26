#include <torch/torch.h>
void PDB2Coords(at::Tensor filenames, at::Tensor coords, at::Tensor res_names, at::Tensor atom_names, int strict);
