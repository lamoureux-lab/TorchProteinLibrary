void cpu_correlationMatrix(     double *d_coords1,  //input: coordinates 1
                                double *d_coords2,  //input: coordinates 2
                                double *T,  //output: T-correlation matrix
                                int *num_atoms, int batch_size, int coords_stride);            //param: number of angles

void cpu_computeR2( double *d_coordinates, int num_atoms, double *R2);

void cpu_transformCoordinates( double *d_coordinates_src, //input: coordinates to transform
                                double *d_coordinates_dst,   //output: transformed coordinates
                                double *d_matrix,            //input: transformation matrix
                                int batch_size, int coords_stride);
