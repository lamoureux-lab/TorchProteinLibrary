template <typename T> void gpu_CoordsTranslateForward(T *coords_src, T *coords_dst, T *translation, int *num_atoms, int batch_size, int atoms_stride);
template <typename T> void gpu_CoordsTranslateBackward(T *grad_coords_output, T *grad_coords_input, T *translation, int *num_atoms, int batch_size, int atoms_stride);

template <typename T> void gpu_CoordsRotateForward(T *coords_src, T *coords_dst, T *rotation, int *num_atoms, int batch_size, int atoms_stride);
template <typename T> void gpu_CoordsRotateBackward(T *grad_coords_output, T *grad_coords_input, T *rotation, int *num_atoms, int batch_size, int atoms_stride);

