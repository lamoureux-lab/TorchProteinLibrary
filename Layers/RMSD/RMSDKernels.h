template <typename T>
void gpu_correlationMatrix( T *d_coords1,  //input: coordinates 1
                            T *d_coords2,  //input: coordinates 2
                            double *TMat,  //output: correlation matrix
                            int *num_atoms, 
                            int batch_size, 
                            int atoms_stride);
template <typename T>
void gpu_computeR2( T *d_coords, int num_atoms, double *R2);
