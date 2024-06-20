#include <torch/extension.h>
void Angles2Coords_forward(     at::Tensor sequences,
                                at::Tensor input_angles, 
                                at::Tensor output_coords,
                                at::Tensor res_names,
                                torch::Tensor res_nums,
                                at::Tensor atom_names,
                                int polymer_type,
                                torch::Tensor chain_names
                        );

void Angles2Coords_backward(    at::Tensor grad_atoms,
                                at::Tensor grad_angles,
                                at::Tensor sequences,
                                at::Tensor input_angles,
                                int polymer_type,
                                torch::Tensor chain_names
                        );
void Angles2Coords_save(    const char* sequence,
                            at::Tensor input_angles, 
                            const char* output_filename,
                            const char mode
                        );
int getSeqNumAtoms( const char *sequence, int polymer_type = 0, torch::Tensor chain_names = {});