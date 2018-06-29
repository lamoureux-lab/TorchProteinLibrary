#include "RMSDKernels.h"


__global__ void gpu_correlationMatrix( double *d_coords1, double *d_coords2, double *R, int *num_atoms, int coords_stride){
    uint batch_idx = blockIdx.x;
    uint i = threadIdx.x;
    uint j = threadIdx.y;
    int r_index = 9*batch_idx + 3*i+j;
    int n_atoms = num_atoms[batch_idx];
    double *coords1 = d_coords1 + batch_idx*coords_stride;
    double *coords2 = d_coords2 + batch_idx*coords_stride;
    
    R[r_index] = 0.0;
    for(int k=0; k<n_atoms; k++){
        R[r_index] += coords1[3*k + i]*coords2[3*k + j];
    }
    
}
__global__ void gpu_TMatrix( double *d_R, double *d_T){
    uint batch_idx = blockIdx.x;
    double *R = d_R + batch_idx*9;
    double *T = d_T + batch_idx*16;

    T[0] = R[0]+R[4]+R[8];      T[1] = R[5]-R[7];           T[2] = R[6]-R[2];            T[3] = R[1]-R[3];
    T[4] = R[5]-R[7];           T[5] = R[0]-R[4]-R[8];      T[6] = R[1]+R[3];            T[7] = R[2]+R[6];
    T[8] = R[6]-R[2];           T[9] = R[1]+R[3];           T[10] = -R[0]+R[4]-R[8];     T[11] = R[5]+R[7];
    T[12] = R[1]-R[3];          T[13] = R[2]+R[6];          T[14] = R[5]+R[7];           T[15] = -R[0]-R[4]+R[8];
}

__global__ void gpu_computeR2( double *d_coordinates, int num_atoms, double *R2){
    int dim_index = threadIdx.x;
    R2[dim_index]=0.0;
    for(int i=0;i<num_atoms;i++){
        R2[dim_index]+=d_coordinates[3*i+dim_index]*d_coordinates[3*i+dim_index];
    }
}

__device__ void mat33Vec3Mul(double *d_m, double *d_v, double *dst){
	if(dst == d_v){
		double tmp[3];
		for(int i=0;i<3;i++){
			tmp[i] = 0.0;
			for(int j=0;j<3;j++){
				tmp[i] += d_m[i*3+j]*d_v[j];
			}
		}
		memcpy(dst, tmp, 3*sizeof(double));
	}else{
		for(int i=0;i<3;i++){
			dst[i] = 0.0;
			for(int j=0;j<3;j++){
				dst[i] += d_m[i*3+j]*d_v[j];
			}
		}
	}
}

__global__ void gpu_transformCoordinates( double *d_coordinates_src, double *d_coordinates_dst, double *d_matrix, int atoms_stride){
    int atom_idx = blockIdx.x;
    int batch_idx = threadIdx.x;
    double *coordinates_src = d_coordinates_src + batch_idx*atoms_stride*3;
    double *coordinates_dst = d_coordinates_dst + batch_idx*atoms_stride*3;
    double *matrix = d_matrix + 9*batch_idx;

    mat33Vec3Mul(matrix, coordinates_src + 3*atom_idx, coordinates_dst + 3*atom_idx);
}

void cpu_correlationMatrix(     double *d_coords1,  //input: coordinates 1
                                double *d_coords2,  //input: coordinates 2
                                double *T,  //output: T-correlation matrix
                                int *num_atoms, int batch_size, int coords_stride){
    double *R ;
    cudaMalloc( &R, batch_size*9*sizeof(double));
    dim3 coords_dim(3, 3, 1);
    gpu_correlationMatrix<<<batch_size, coords_dim>>>(d_coords1, d_coords2, R, num_atoms, coords_stride);
    gpu_TMatrix<<<batch_size,1>>>(R, T);
    cudaFree(R);
}

void cpu_computeR2( double *d_coordinates, int num_atoms, double *R2){
    gpu_computeR2<<<1,3>>>( d_coordinates, num_atoms, R2);
}


void cpu_transformCoordinates( double *d_coordinates_src, //input: coordinates to transform
                                double *d_coordinates_dst,   //output: transformed coordinates
                                double *d_matrix,            //input: transformation matrix
                                int batch_size, int coords_stride){
    int max_num_atoms = coords_stride/3;
    gpu_transformCoordinates<<<max_num_atoms, batch_size>>>(d_coordinates_src, d_coordinates_dst, d_matrix, max_num_atoms);
}


void toGPUTensor(THCState*state, double *cpu_T, THCudaDoubleTensor *gpu_T){
    uint size = 1;
    for(int i=0; i<gpu_T->nDimension; i++)
        size *= gpu_T->size[i];
    cudaMemcpy( THCudaDoubleTensor_data(state, gpu_T), 
                cpu_T,
                size*sizeof(double), cudaMemcpyHostToDevice);
}

void toCPUTensor(THCState*state, THDoubleTensor *cpu_T, THCudaDoubleTensor *gpu_T){
    uint size = 1;
    for(int i=0; i<gpu_T->nDimension; i++)
        size *= gpu_T->size[i];
    cudaMemcpy( THDoubleTensor_data(cpu_T), 
                THCudaDoubleTensor_data(state, gpu_T),
                size*sizeof(double), cudaMemcpyDeviceToHost);
}

THCudaDoubleTensor* toGPU(THCState*state, THDoubleTensor *T){
    THCudaDoubleTensor *gpu_T = THCudaDoubleTensor_newWithSize(state, THDoubleTensor_newSizeOf(T), THDoubleTensor_newStrideOf(T));
    toGPUTensor(state, THDoubleTensor_data(T), gpu_T);
    return gpu_T;
}

THDoubleTensor* fromGPU(THCState*state, THCudaDoubleTensor *T){
    THDoubleTensor *cpu_T = THDoubleTensor_newWithSize(THCudaDoubleTensor_newSizeOf(state, T), THCudaDoubleTensor_newStrideOf(state, T));
    toCPUTensor(state, cpu_T, T);
    return cpu_T;
}

