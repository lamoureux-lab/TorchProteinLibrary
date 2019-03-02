
#define REAL float

void gpu_computeCoordinatesBackbone(REAL *angles, 
                                    REAL *dr, 
                                    REAL *dR_dangle, 
                                    int *length, 
                                    int batch_size, 
                                    int angles_stride);

void gpu_computeDerivativesBackbone(REAL *angles,  
									REAL *atoms,   
									REAL *A,       
									int *length,
									int batch_size,
									int angles_stride);

void gpu_backwardFromCoordsBackbone(REAL *angles,
									REAL *dR_dangle,
									REAL *A,
									int *length,
									int batch_size,
									int angles_stride);