#include "RMSDKernels.h"
#include <cMathCUDAKernels.cu>

template <typename T>
__global__ void cuda_correlationMatrix( T *d_coords1, T *d_coords2, double *RMat, int *num_atoms, int atoms_stride){
    uint batch_idx = blockIdx.x;
    uint i = threadIdx.x;
    uint j = threadIdx.y;
    int r_index = 9*batch_idx + 3*i+j;
    int n_atoms = num_atoms[batch_idx];
    T *coords1 = d_coords1 + batch_idx*atoms_stride*3;
    T *coords2 = d_coords2 + batch_idx*atoms_stride*3;
    
    RMat[r_index] = 0.0;
    for(int k=0; k<n_atoms; k++){
        RMat[r_index] += ((double)coords1[3*k + i]) * ((double)coords2[3*k + j]);
    }
    
}

__global__ void cuda_TMatrix( double *d_RMat, double *d_TMat){
    uint batch_idx = blockIdx.x;
    double *RMat = d_RMat + batch_idx*9;
    double *TMat = d_TMat + batch_idx*16;

    TMat[0] = RMat[0]+RMat[4]+RMat[8];   TMat[1] = RMat[5]-RMat[7];           TMat[2] = RMat[6]-RMat[2];            TMat[3] = RMat[1]-RMat[3];
    TMat[4] = RMat[5]-RMat[7];           TMat[5] = RMat[0]-RMat[4]-RMat[8];   TMat[6] = RMat[1]+RMat[3];            TMat[7] = RMat[2]+RMat[6];
    TMat[8] = RMat[6]-RMat[2];           TMat[9] = RMat[1]+RMat[3];           TMat[10] = -RMat[0]+RMat[4]-RMat[8];  TMat[11] = RMat[5]+RMat[7];
    TMat[12] = RMat[1]-RMat[3];          TMat[13] = RMat[2]+RMat[6];          TMat[14] = RMat[5]+RMat[7];           TMat[15] = -RMat[0]-RMat[4]+RMat[8];
}

template <typename T>
__global__ void cuda_computeR2( T *d_coordinates, int num_atoms, double *R2){
    int dim_index = threadIdx.x;
    R2[dim_index] = 0.0;
    for(int i=0; i<num_atoms; i++){
        R2[dim_index] += ((double)d_coordinates[3*i+dim_index]) * ((double)d_coordinates[3*i+dim_index]);
    }
}

template <typename T>
void gpu_correlationMatrix(T *d_coords1, T *d_coords2, double *TMat, int *num_atoms, int batch_size, int atoms_stride){
    double *RMat;
    cudaMalloc( &RMat, batch_size*9*sizeof(double));
    dim3 coords_dim(3, 3, 1);
    cuda_correlationMatrix<T><<<batch_size, coords_dim>>>(d_coords1, d_coords2, RMat, num_atoms, atoms_stride);
    cuda_TMatrix<<<batch_size,1>>>(RMat, TMat);
    cudaFree(RMat);
}

template <typename T>
void gpu_computeR2( T *d_coordinates, int num_atoms, double *R2){
    cuda_computeR2<T><<<1,3>>>( d_coordinates, num_atoms, R2);
}


template void gpu_correlationMatrix<float>(float*, float*, double*, int*, int, int);
template void gpu_correlationMatrix<double>(double*, double*, double*, int*, int, int);

template void gpu_computeR2<float>(float*, int, double*);
template void gpu_computeR2<double>(double*, int, double*);