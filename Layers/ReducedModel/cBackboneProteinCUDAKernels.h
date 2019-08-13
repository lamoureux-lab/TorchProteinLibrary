
#define REAL float
template <typename T>
void gpu_computeCoordinatesBackbone(T *angles, 
                                    T *dr, 
                                    T *dR_dangle, 
                                    int *length, 
                                    int batch_size, 
                                    int angles_stride);
template <typename T>
void gpu_computeDerivativesBackbone(T *angles,  
									T *atoms,   
									T *A,       
									int *length,
									int batch_size,
									int angles_stride);
template <typename T>
void gpu_backwardFromCoordsBackbone(T *angles,
									T *dR_dangle,
									T *A,
									int *length,
									int batch_size,
									int angles_stride);