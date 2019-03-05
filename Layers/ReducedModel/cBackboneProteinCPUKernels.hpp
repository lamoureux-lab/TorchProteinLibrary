
void cpu_computeCoordinatesBackbone(    double *angles, 
                                    double *dr, 
                                    double *dR_dangle, 
                                    int *length, 
                                    int batch_size, 
                                    int angles_stride);

void cpu_computeDerivativesBackbone(    double *angles,  
                                        double *dR_dangle,   
                                        double *A,       
                                        int *length,
                                        int batch_size,
                                        int angles_stride);

void cpu_backwardFromCoordsBackbone(    double *gradInput,
                                        double *gradOutput,
                                        double *dR_dangle,
                                        int *length,
                                        int batch_size,
                                        int angles_stride);