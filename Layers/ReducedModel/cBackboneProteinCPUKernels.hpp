template <typename T>
void cpu_computeCoordinatesBackbone(    T *angles, 
                                        T *param,
										T *dr, 
										T *dR_dangle, 
										int *length, 
										int batch_size, 
										int angles_stride);
template <typename T>
void cpu_computeDerivativesBackbone(    T *angles,  
                                        T *param,
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

template <typename T>
void cpu_computeDerivativesParam(   T *angles,
                                    T *param,
                                    T *dR_dparam,   
                                    T *A,       
                                    int *length,
                                    int batch_size,
                                    int angles_stride);

template <typename T>
void cpu_backwardFromCoordsParam(   T *gradParam,
                                    T *gradOutput,
                                    T *dR_dparam,
                                    int *length,
                                    int batch_size,
                                    int angles_stride);