#include <THC/THC.h>

void gpu_computeCoords2Volume(	double *coords,
                                int *num_atoms_of_type,
							    int *offsets, 
								float *volume,
								int spatial_dim,
                                int num_atom_types,
								float res);

void gpu_computeVolume2Coords(	double *coords,
								double* grad,
                                int *num_atoms_of_type,
							    int *offsets, 
								float *volume,
								int spatial_dim,
                                int num_atom_types,
								float res);

void gpu_selectFromTensor(	float *features, int num_features, 
							float* volume, int spatial_dim, 
							float *coords, int num_atoms, int max_num_atoms, 
							float res);