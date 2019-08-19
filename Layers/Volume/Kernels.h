#include <THC/THC.h>

template <typename T>
void gpu_computeCoords2Volume(	T *coords,
                                int *num_atoms_of_type,
							    int *offsets, 
								T *volume,
								int spatial_dim,
                                int num_atom_types,
								float res);

template <typename T>
void gpu_computeVolume2Coords(	T *coords,
								T* grad,
                                int *num_atoms_of_type,
							    int *offsets, 
								T *volume,
								int spatial_dim,
                                int num_atom_types,
								float res);

void gpu_selectFromTensor(	float *features, int num_features, 
							float* volume, int spatial_dim, 
							float *coords, int num_atoms, int max_num_atoms, 
							float res);