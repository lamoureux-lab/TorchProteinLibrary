#include <THC/THC.h>

void gpu_computeCoords2Volume(	double *coords,
                                uint *num_atoms_of_type,
							    uint *offsets, 
								float *volume,
								uint spatial_dim,
                                uint num_atom_types,
								float res);

void gpu_computeVolume2Coords(	double *coords,
								double* grad,
                                uint *num_atoms_of_type,
							    uint *offsets, 
								float *volume,
								uint spatial_dim,
                                uint num_atom_types,
								float res);