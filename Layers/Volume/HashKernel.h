template <typename T>
void gpu_computeCoords2Volume(	T *coords,
                                int num_atoms,
								T *volume,
								int spatial_dim,
								float res, int d,
								long *gridParticleHash,
								long *gridParticleIndex,
								long *cellStart,
								long *cellStop,
								T *sortedPos);

template <typename T>
void gpu_computeVolume2Coords(	T *coords,
								T* grad,
                                int num_atoms, 
								T *volume,
								int spatial_dim,
								float res,
								int d);