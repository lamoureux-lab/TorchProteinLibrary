template <typename T>
void gpu_computeCoordinatesBackbone(T *angles, 
									T *param, 
									T *atoms, 
									T *A, 
									int *length, 
									int batch_size, 
									int angles_stride);
template <typename T>
void gpu_computeDerivativesBackbone(T *angles, 
									T *param, 
									T *dR_dangle, 
									T *A, 
									int *length, 
									int batch_size, 
									int angles_stride);
template <typename T>
void gpu_backwardFromCoordsBackbone(T *angles, 
									T *dr, 
									T *dR_dangle, 
									int *length, 
									int batch_size, 
									int angles_stride);

template <typename T>
void gpu_computeDerivativesParam(	T *angles, 
									T *param, 
									T *dR_dparam, 
									T *A, 
									int *length, 
									int batch_size, 
									int angles_stride);

template <typename T>
void gpu_backwardFromCoordsParam(	T *gradParam, 
									T *dr, 
									T *dR_dparam, 
									int *length, 
									int batch_size, 
									int angles_stride);