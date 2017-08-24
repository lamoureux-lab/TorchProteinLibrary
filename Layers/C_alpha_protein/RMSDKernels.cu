#include "RMSDKernels.h"



__global__ void gpu_computeCentroid( float *d_coords, float *d_centroid, int L ){
    uint dim_index = blockIdx.x;
    int n_atoms = L + 1;
    d_centroid[dim_index]=0.0;
    for(int i=0;i<n_atoms;i++){
        // printf("Coords[%d]=%f",3*i+dim_index,d_coords[3*i+dim_index]);
        d_centroid[dim_index]+=d_coords[3*i+dim_index];
    }
    d_centroid[dim_index]/=n_atoms;
    // printf("Centroid[%d]=%f, n atoms = %d\n",dim_index,d_centroid[dim_index],n_atoms);
}

__global__ void gpu_centerCoords( float *d_coords_src, float *d_centroid, float *d_coords_dst){
    uint dim_index = blockIdx.x;
    uint atom_index = threadIdx.x;
    d_coords_dst[3*atom_index + dim_index] = d_coords_src[3*atom_index + dim_index] - d_centroid[dim_index];
    // printf("Centroid[%d]=%f\n",dim_index,d_centroid[dim_index]);
}


__global__ void gpu_correlationMatrix( float *d_coords1, float *d_coords2, double *R, int L){
    uint i = blockIdx.x;
    uint j = threadIdx.x;
    int r_index = 3*i+j;
    int n_atoms = L+1;
    // float norm_const = n_atoms*n_atoms; //without it the values will overflow (loose precision) and the final rotation will be incorrect
    float norm_const = 1.0; //without it the values will overflow (loose precision) and the final rotation will be incorrect
    R[r_index] = 0.0;
    for(int k=0; k<n_atoms; k++){
        R[r_index] += double(d_coords1[3*k + i])*double(d_coords2[3*k + j])/norm_const;
    }
    // printf("R-Matrix[%d, %d]=%f\n",i,j,R[r_index]);
}
__global__ void gpu_TMatrix( double *R, double *T){
    T[0] = R[0]+R[4]+R[8];      T[1] = R[5]-R[7];           T[2] = R[6]-R[2];            T[3] = R[1]-R[3];
    T[4] = R[5]-R[7];           T[5] = R[0]-R[4]-R[8];      T[6] = R[1]+R[3];            T[7] = R[2]+R[6];
    T[8] = R[6]-R[2];           T[9] = R[1]+R[3];           T[10] = -R[0]+R[4]-R[8];     T[11] = R[5]+R[7];
    T[12] = R[1]-R[3];          T[13] = R[2]+R[6];          T[14] = R[5]+R[7];           T[15] = -R[0]-R[4]+R[8];
}

__global__ void gpu_constructRotMatrix( float *d_eigenvectors, float *d_eigenvalues, float *U, float *eig){
    // getting maximal eigenvalue
    int max_ind = 0;
    float max_eig = d_eigenvalues[max_ind];
    for(int i=max_ind; i<4; i++){
        if(max_eig<d_eigenvalues[i]){
            max_ind = i;
            max_eig = d_eigenvalues[max_ind];
        }
    }
    // getting corresponding eigenvector
    float q[4];
    for(int j=0; j<4; j++){
        q[j] = d_eigenvectors[4*max_ind + j];
    }

    //constructing rotation matrix
    U[0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
    U[1] = 2.0*(q[1]*q[2] - q[0]*q[3]);
    U[2] = 2.0*(q[1]*q[3] + q[0]*q[2]);

    U[3] = 2.0*(q[1]*q[2] + q[0]*q[3]);
    U[4] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
    U[5] = 2.0*(q[2]*q[3] - q[0]*q[1]);

    U[6] = 2.0*(q[1]*q[3] - q[0]*q[2]);
    U[7] = 2.0*(q[2]*q[3] + q[0]*q[1]);
    U[8] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];

    *eig = max_eig;
}

__global__ void gpu_computeR2( float *d_coordinates, int L, double *R2){
    int dim_index = threadIdx.x;
    int n_atoms = L + 1;
    R2[dim_index]=0.0;
    for(int i=0;i<n_atoms;i++){
        R2[dim_index]+=d_coordinates[3*i+dim_index]*d_coordinates[3*i+dim_index];
    }
}

__device__ void mat33Vec3Mul(float *d_m, float *d_v, float *dst){
	if(dst == d_v){
		float tmp[3];
		for(int i=0;i<3;i++){
			tmp[i] = 0.0;
			for(int j=0;j<3;j++){
				tmp[i] += d_m[i*3+j]*d_v[j];
			}
		}
		memcpy(dst, tmp, 3*sizeof(float));
	}else{
		for(int i=0;i<3;i++){
			dst[i] = 0.0;
			for(int j=0;j<3;j++){
				dst[i] += d_m[i*3+j]*d_v[j];
			}
		}
	}
}

__global__ void gpu_transformCoordinates( float *d_coordinates_src, float *d_coordinates_dst, float *d_matrix, int L){
    int atom_idx = threadIdx.x;
    mat33Vec3Mul(d_matrix, d_coordinates_src+3*atom_idx, d_coordinates_dst+3*atom_idx);
}
__global__ void gpu_mulAndSub( float *d_coordinates_dst, float *d_coordinates1, float *d_coordinates2, float mult, int L){
    int atom_idx = threadIdx.x;
    int dim_index = blockIdx.x;
    d_coordinates_dst[3*atom_idx + dim_index] = mult*(d_coordinates1[3*atom_idx + dim_index] - d_coordinates2[3*atom_idx + dim_index]);
}


void cpu_computeCentroid(   float *d_coords,  //input: coordinates 
                            float *d_centroid,  //output: centroid 
                            int L ){            //param: number of angles
    gpu_computeCentroid<<<3, 1>>>(d_coords, d_centroid, L);
}

void cpu_centerCoords(      float *d_coords_src,  //input: initial coordinates 
                            float *d_centroid,  //input: centroid 
                            float *d_coords_dst,  //output: centered coordinates 
                            int L ){            //param: number of angles
    gpu_centerCoords<<<3, L+1>>>(d_coords_src, d_centroid, d_coords_dst);
}

void cpu_correlationMatrix(     float *d_coords1,  //input: coordinates 1
                                float *d_coords2,  //input: coordinates 2
                                double *T,  //output: T-correlation matrix
                                int L ){            //param: number of angles
    double *R ;
    cudaMalloc( &R, 9*sizeof(double));
    gpu_correlationMatrix<<<3, 3>>>(d_coords1, d_coords2, R, L);
    gpu_TMatrix<<<1,1>>>(R, T);
    cudaFree(R);
}

float cpu_constructRotMatrix(    float *d_eigenvectors,  //input: T-eigenvectors
                                float *d_eigenvalues,   //input: T-eigenvalues
                                float *U){              //output: rotation matrix
    float eig;
    gpu_constructRotMatrix<<<1,1>>>(d_eigenvectors, d_eigenvalues, U, &eig);
    return eig;
}

void cpu_computeR2( float *d_coordinates, int L, double *R2){
    gpu_computeR2<<<1,3>>>( d_coordinates, L, R2);
}


void cpu_transformCoordinates( float *d_coordinates_src, //input: coordinates to transform
                                float *d_coordinates_dst,   //output: transformed coordinates
                                float *d_matrix,            //input: transformation matrix
                                int L){                     //param: number of angles
    gpu_transformCoordinates<<<1, L+1>>>(d_coordinates_src, d_coordinates_dst, d_matrix, L);
}

void cpu_mulAndSub( float *d_coordinates_dst,
                    float *d_coordinates1,
                    float *d_coordinates2,
                    float mult,
                    int L){
    gpu_mulAndSub<<<3,L+1>>>(d_coordinates_dst, d_coordinates1, d_coordinates2, mult, L);
}


THCudaTensor* toGPU(THCState*state, THFloatTensor *T){
    THCudaTensor *gpu_T = THCudaTensor_newWithSize(state, THFloatTensor_newSizeOf(T), THFloatTensor_newStrideOf(T));
    uint size = 1;
    for(int i=0; i<T->nDimension; i++)
        size *= T->size[i];
    cudaMemcpy( THCudaTensor_data(state, gpu_T), 
                THFloatTensor_data(T), 
                size*sizeof(float), cudaMemcpyHostToDevice);
    return gpu_T;
}

THFloatTensor* fromGPU(THCState*state, THCudaTensor *T){
    THFloatTensor *cpu_T = THFloatTensor_newWithSize(THCudaTensor_newSizeOf(state, T), THCudaTensor_newStrideOf(state, T));
    uint size = 1;
    for(int i=0; i<T->nDimension; i++)
        size *= T->size[i];
    // printf("Size = %d", size);
    cudaMemcpy( THFloatTensor_data(cpu_T), 
                THCudaTensor_data(state, T), 
                size*sizeof(float), cudaMemcpyDeviceToHost);
    return cpu_T;
}

THDoubleTensor* fromGPUDouble(THCState*state, THCudaDoubleTensor *T){
    THDoubleTensor *cpu_T = THDoubleTensor_newWithSize(THCudaDoubleTensor_newSizeOf(state, T), THCudaDoubleTensor_newStrideOf(state, T));
    uint size = 1;
    for(int i=0; i<T->nDimension; i++)
        size *= T->size[i];
    // printf("Size = %d", size);
    cudaMemcpy( THDoubleTensor_data(cpu_T), 
                THCudaDoubleTensor_data(state, T), 
                size*sizeof(double), cudaMemcpyDeviceToHost);
    return cpu_T;
}

void toGPUTensor(THCState*state, float *cpu_T, THCudaTensor *gpu_T){
    uint size = 1;
    for(int i=0; i<gpu_T->nDimension; i++)
        size *= gpu_T->size[i];
    cudaMemcpy( THCudaTensor_data(state, gpu_T), 
                cpu_T,
                size*sizeof(float), cudaMemcpyHostToDevice);
}

void toCPUTensor(THCState*state, THFloatTensor *cpu_T, THCudaTensor *gpu_T){
    uint size = 1;
    for(int i=0; i<gpu_T->nDimension; i++)
        size *= gpu_T->size[i];
    cudaMemcpy( THFloatTensor_data(cpu_T), 
                THCudaTensor_data(state, gpu_T),
                size*sizeof(float), cudaMemcpyDeviceToHost);
}