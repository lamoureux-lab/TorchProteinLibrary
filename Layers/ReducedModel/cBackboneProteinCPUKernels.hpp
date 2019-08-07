template <typename T>
void cpu_computeCoordinatesBackbone(    T *angles, 
										T *dr, 
										T *dR_dangle, 
										int *length, 
										int batch_size, 
										int angles_stride);
template <typename T>
void cpu_computeDerivativesBackbone(    T *angles,  
                                        T *dR_dangle,   
                                        T *A,       
                                        int *length,
                                        int batch_size,
                                        int angles_stride);
template <typename T>
void cpu_backwardFromCoordsBackbone(    T *gradInput,
                                        T *gradOutput,
                                        T *dR_dangle,
                                        int *length,
                                        int batch_size,
                                        int angles_stride);