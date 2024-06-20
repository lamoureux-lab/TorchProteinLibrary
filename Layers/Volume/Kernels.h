/*
template <typename T>
void gpu_computeCoords2Volume(	T *coords,
                                int num_atoms,
								T *volume,
								int spatial_dim,
								float res);

template <typename T>
void gpu_computeVolume2Coords(	T *coords,
								T* grad,
                                int num_atoms, 
								T *volume,
								int spatial_dim,
								float res);
*/
void gpu_coordSelect(	float *features, int num_features, 
						float* volume, int spatial_dim, 
						float *coords, int num_atoms, int max_num_atoms, 
						float res);

void gpu_coordSelectGrad(	float *gradOutput, int num_features, 
							float* gradInput, int spatial_dim, 
							float *coords, int num_atoms, int max_num_atoms, 
							float res);