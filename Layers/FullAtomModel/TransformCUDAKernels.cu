#include <cMathCUDAKernels.cu>

template <typename T>
__global__ void cuda_CoordsTranslateForward( T *coords_src, T *coords_dst, T *translation, int *num_atoms, int atoms_stride){
    int atom_idx = blockIdx.x;
    int batch_idx = threadIdx.x;
    T *r_src = coords_src + batch_idx*atoms_stride*3;
    T *r_dst = coords_dst + batch_idx*atoms_stride*3;
    T *tr = translation + 3*batch_idx;

    if(atom_idx<num_atoms[batch_idx])
        vec3Plus<T>(r_src+3*atom_idx, tr, r_dst+3*atom_idx);
}

template <typename T>
void gpu_CoordsTranslateForward(T *coords_src, T *coords_dst, T *translation, int *num_atoms, int batch_size, int atoms_stride){
    cuda_CoordsTranslateForward<T><<<atoms_stride, batch_size>>>(coords_src, coords_dst, translation, num_atoms, atoms_stride);
}


template <typename T>
__global__ void cuda_CoordsTranslateBackward( T *grad_coords_output, T *grad_coords_input, T *translation, int *num_atoms, int atoms_stride){
    int atom_idx = blockIdx.x;
    int batch_idx = threadIdx.x;
    T *r_grad_out = grad_coords_output + batch_idx*atoms_stride*3;
    T *r_grad_in = grad_coords_input + batch_idx*atoms_stride*3;
    T *tr = translation + 3*batch_idx;

    if(atom_idx<num_atoms[batch_idx])
        setVec3<T>(r_grad_out+3*atom_idx, r_grad_in+3*atom_idx);
        
}

template <typename T>
void gpu_CoordsTranslateBackward(T *grad_coords_output, T *grad_coords_input, T *translation, int *num_atoms, int batch_size, int atoms_stride){
    cuda_CoordsTranslateBackward<T><<<atoms_stride, batch_size>>>(grad_coords_output, grad_coords_input, translation, num_atoms, atoms_stride);
}


template <typename T>
__global__ void cuda_CoordsRotateForward( T *coords_src, T *coords_dst, T *rotation, int *num_atoms, int atoms_stride){
    int atom_idx = blockIdx.x;
    int batch_idx = threadIdx.x;
    T *r_src = coords_src + batch_idx*atoms_stride*3;
    T *r_dst = coords_dst + batch_idx*atoms_stride*3;
    T *R = rotation + 9*batch_idx;

    if(atom_idx<num_atoms[batch_idx])
        mat33Vec3Mul<T>(R, r_src + 3*atom_idx, r_dst + 3*atom_idx);
}

template <typename T>
void gpu_CoordsRotateForward(T *coords_src, T *coords_dst, T *rotation, int *num_atoms, int batch_size, int atoms_stride){
    cuda_CoordsRotateForward<T><<<atoms_stride, batch_size>>>(coords_src, coords_dst, rotation, num_atoms, atoms_stride);
}

template <typename T>
__global__ void cuda_CoordsRotateBackward( T *coords_src, T *coords_dst, T *rotation, int *num_atoms, int atoms_stride){
    int atom_idx = blockIdx.x;
    int batch_idx = threadIdx.x;
    T *r_src = coords_src + batch_idx*atoms_stride*3;
    T *r_dst = coords_dst + batch_idx*atoms_stride*3;
    T *R = rotation + 9*batch_idx;
    T RT[9]; //transposed rotation matrix
    mat33Transpose<T>(R, RT);

    if(atom_idx<num_atoms[batch_idx])
        mat33Vec3Mul<T>(RT, r_src + 3*atom_idx, r_dst + 3*atom_idx);
}

template <typename T>
void gpu_CoordsRotateBackward(T *grad_coords_output, T *grad_coords_input, T *rotation, int *num_atoms, int batch_size, int atoms_stride){
    cuda_CoordsRotateBackward<T><<<atoms_stride, batch_size>>>(grad_coords_output, grad_coords_input, rotation, num_atoms, atoms_stride);
}

template void gpu_CoordsTranslateForward<float>(float*, float*, float*, int*, int, int);
template void gpu_CoordsTranslateBackward<float>(float*, float*, float*, int*, int, int);
template void gpu_CoordsTranslateForward<double>(double*, double*, double*, int*, int, int);
template void gpu_CoordsTranslateBackward<double>(double*, double*, double*, int*, int, int);

template void gpu_CoordsRotateForward<float>(float*, float*, float*, int*, int, int);
template void gpu_CoordsRotateBackward<float>(float*, float*, float*, int*, int, int);
template void gpu_CoordsRotateForward<double>(double*, double*, double*, int*, int, int);
template void gpu_CoordsRotateBackward<double>(double*, double*, double*, int*, int, int);