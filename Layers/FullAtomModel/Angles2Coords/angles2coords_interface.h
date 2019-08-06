#include <torch/extension.h>
void Angles2Coords_forward(     at::Tensor sequences,
                                at::Tensor input_angles, 
                                at::Tensor output_coords,
                                at::Tensor res_names,
                                at::Tensor atom_names
                        );

void Angles2Coords_backward(    at::Tensor grad_atoms,
                                at::Tensor grad_angles,
                                at::Tensor sequences,
                                at::Tensor input_angles
                        );
void Angles2Coords_save(    const char* sequence,
                            at::Tensor input_angles, 
                            const char* output_filename,
                            const char mode
                        );
int getSeqNumAtoms( const char *sequence);