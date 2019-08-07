#include <torch/extension.h>
void Coords2TypedCoords_forward(    torch::Tensor input_coords, 
                                    torch::Tensor res_names,
                                    torch::Tensor atom_names,
                                    torch::Tensor input_num_atoms,
                                    torch::Tensor output_coords,
                                    torch::Tensor output_num_atoms_of_type,
                                    torch::Tensor output_offsets,
                                    torch::Tensor output_atom_indexes
                                );
void Coords2TypedCoords_backward(   torch::Tensor grad_typed_coords,
                                    torch::Tensor grad_flat_coords,
                                    torch::Tensor num_atoms_of_type,
                                    torch::Tensor offsets,
                                    torch::Tensor atom_indexes
                                );