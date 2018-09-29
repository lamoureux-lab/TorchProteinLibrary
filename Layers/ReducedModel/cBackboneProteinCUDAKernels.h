
#define REAL float

void cpu_computeCoordinatesBackbone(REAL *angles, 
                                    REAL *dr, 
                                    REAL *dR_dangle, 
                                    int *length, 
                                    int batch_size, 
                                    int angles_stride);

void cpu_computeDerivativesBackbone(REAL *angles,  
									REAL *atoms,   
									REAL *A,       
									int *length,
									int batch_size,
									int angles_stride);

void cpu_backwardFromCoordsBackbone(REAL *angles,
									REAL *dR_dangle,
									REAL *A,
									int *length,
									int batch_size,
									int angles_stride);