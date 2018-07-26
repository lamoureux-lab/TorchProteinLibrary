#include "VolumeConv.h"
#include <cufft.h>

#define WARP_SIZE 32

__global__ void conjMul(cufftComplex *c_volume1, cufftComplex *c_volume2, cufftComplex *c_output, int batch_size, int volume_size, bool conj){
	uint batch_idx = blockIdx.x;
    uint warp_idx = blockIdx.y;
    uint thread_idx = threadIdx.x;
    uint vol = volume_size*volume_size*volume_size;
    uint memory_idx = batch_idx*vol + warp_idx*WARP_SIZE + thread_idx;

    if( (warp_idx*WARP_SIZE + thread_idx) >= vol) //out of volume
        return;
    REAL re, im;
    if(conj){
        re = c_volume1[memory_idx].x * c_volume2[memory_idx].x + c_volume1[memory_idx].y * c_volume2[memory_idx].y;
        im = c_volume1[memory_idx].x * c_volume2[memory_idx].y + c_volume1[memory_idx].y * c_volume2[memory_idx].x;
    }else{
        re = c_volume1[memory_idx].x * c_volume2[memory_idx].x - c_volume1[memory_idx].y * c_volume2[memory_idx].y;
        im = c_volume1[memory_idx].x * c_volume2[memory_idx].y + c_volume1[memory_idx].y * c_volume2[memory_idx].x;
    }

    c_output[memory_idx].x = re;
    c_output[memory_idx].y = im;

}

__global__ void mulInPlace(REAL *d_volume1, REAL *d_volume2, int batch_size, int volume_size){
	uint batch_idx = blockIdx.x;
    uint warp_idx = blockIdx.y;
    uint thread_idx = threadIdx.x;
    uint vol = volume_size*volume_size*volume_size;
    uint memory_idx = batch_idx*vol + warp_idx*WARP_SIZE + thread_idx;

    if( (warp_idx*WARP_SIZE + thread_idx) >= vol) //out of volume
        return;

    d_volume1[memory_idx] *= d_volume2[memory_idx];
}

void cpu_VolumeConv(REAL *d_volume1,  REAL *d_volume2,  REAL *d_output, int batch_size, int volume_size){
    cufftHandle plan_fwd, plan_bwd;
    cufftComplex *c_volume1, *c_volume2, *c_output;
    cudaMalloc((void**)&c_volume1, sizeof(cufftComplex)*batch_size*volume_size*volume_size*volume_size);
    cudaMalloc((void**)&c_volume2, sizeof(cufftComplex)*batch_size*volume_size*volume_size*volume_size);
    cudaMalloc((void**)&c_output, sizeof(cufftComplex)*batch_size*volume_size*volume_size*volume_size);

    int dimensions [volume_size, volume_size, volume_size];
    cufftPlanMany(&plan_fwd, 3, dimensions, NULL, 0, 0, NULL, 0, 0,
                    CUFFT_R2C, batch_size);
    cufftPlanMany(&plan_bwd, 3, dimensions, NULL, 0, 0, NULL, 0, 0,
                    CUFFT_C2R, batch_size);
    cufftExecR2C(plan_fwd, d_volume1, c_volume1);
    cufftExecR2C(plan_fwd, d_volume2, c_volume2);

    dim3 dim_special(batch_size, volume_size*volume_size*volume_size/WARP_SIZE + 1);
	conjMul<<<dim_special, WARP_SIZE>>>(c_volume1, c_volume2, c_output, batch_size, volume_size, false);

    cufftExecC2R(plan_bwd, c_output, d_output);
    
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_bwd);
    cudaFree(c_volume1);
    cudaFree(c_volume2);
    cudaFree(c_output);

}

void cpu_VolumeConvGrad( REAL *d_gradOutput, REAL *d_volume1, REAL *d_volume2, REAL *d_gradVolume1, int batch_size, int volume_size, bool conjugate){
    cufftHandle plan_fwd, plan_bwd;
    cufftComplex *c_volume1, *c_gradOutput, *c_gradVolume1;
    cudaMalloc((void**)&c_volume1, sizeof(cufftComplex)*batch_size*volume_size*volume_size*volume_size);
    cudaMalloc((void**)&c_gradOutput, sizeof(cufftComplex)*batch_size*volume_size*volume_size*volume_size);
    cudaMalloc((void**)&c_gradVolume1, sizeof(cufftComplex)*batch_size*volume_size*volume_size*volume_size);

    int dimensions [volume_size, volume_size, volume_size];
    cufftPlanMany(&plan_fwd, 3, dimensions, NULL, 0, 0, NULL, 0, 0,
                    CUFFT_R2C, batch_size);
    cufftPlanMany(&plan_bwd, 3, dimensions, NULL, 0, 0, NULL, 0, 0,
                    CUFFT_C2R, batch_size);
    cufftExecR2C(plan_fwd, d_volume1, c_volume1);
    cufftExecR2C(plan_fwd, d_gradOutput, c_gradOutput);

    dim3 dim_special(batch_size, volume_size*volume_size*volume_size/WARP_SIZE + 1);
    conjMul<<<dim_special, WARP_SIZE>>>(c_gradOutput, c_volume1, c_gradVolume1, batch_size, volume_size, conjugate);

    cufftExecC2R(plan_bwd, c_gradVolume1, d_gradVolume1);

    mulInPlace<<<dim_special, WARP_SIZE>>>(d_gradVolume1, d_volume2, batch_size, volume_size);
    
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_bwd);
    cudaFree(c_volume1);
    cudaFree(c_gradOutput);
    cudaFree(c_gradVolume1);

}
