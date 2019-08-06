#include <torch/extension.h>
void Coords2TypedCoords_forward(    at::Tensor input_coords, 
                                    at::Tensor res_names,
                                    at::Tensor atom_names,
                                    at::Tensor input_num_atoms,
                                    at::Tensor output_coords,
                                    at::Tensor output_num_atoms_of_type,
                                    at::Tensor output_offsets,
                                    at::Tensor output_atom_indexes
                                );
void Coords2TypedCoords_backward(   at::Tensor grad_typed_coords,
                                    at::Tensor grad_flat_coords,
                                    at::Tensor num_atoms_of_type,
                                    at::Tensor offsets,
                                    at::Tensor atom_indexes
                                );