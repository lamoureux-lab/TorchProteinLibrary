#include <torch/extension.h>
void PDB2CoordsOrdered( torch::Tensor filenames, torch::Tensor coords, torch::Tensor chain_names, torch::Tensor res_names, 
                        torch::Tensor res_nums, torch::Tensor atom_names, torch::Tensor atom_mask, torch::Tensor num_atoms, int polymer_type);
void PDB2CoordsUnordered(torch::Tensor filenames, torch::Tensor coords, torch::Tensor chain_names, torch::Tensor res_names, torch::Tensor res_nums, torch::Tensor atom_names, torch::Tensor num_atoms);
