#include "VolumeConv.h"
#include <cufft.h>
#include <stdio.h>

#define gpuFFTErrchk(err) { __cufftSafeCall(err, __FILE__, __LINE__); }
#define gpuErrchk(err) { gpuAssert(err, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
        fprintf(stderr, "CUFFT error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, \
                                    _cudaGetErrorEnum(err)); \
        cudaDeviceReset(); exit(1); \
    }
}

#define WARP_SIZE 32

__global__ void conjMul(cufftComplex *c_volume1, cufftComplex *c_volume2, cufftComplex *c_output, int batch_size, int volume_size, bool conj){
	uint batch_idx = blockIdx.x;
    uint warp_idx = blockIdx.y;
    uint thread_idx = threadIdx.x;
    int reduced_volume_size = (volume_size/2 + 1);
    uint vol = volume_size*volume_size*reduced_volume_size;
    uint full_vol = volume_size*volume_size*volume_size;
    uint memory_idx = batch_idx*vol + warp_idx*WARP_SIZE + thread_idx;

    if( (warp_idx*WARP_SIZE + thread_idx) >= vol) //out of volume
        return;
    REAL re, im;
    if(conj){
        re = c_volume1[memory_idx].x * c_volume2[memory_idx].x + c_volume1[memory_idx].y * c_volume2[memory_idx].y;
        im = -c_volume1[memory_idx].x * c_volume2[memory_idx].y + c_volume1[memory_idx].y * c_volume2[memory_idx].x;
    }else{
        re = c_volume1[memory_idx].x * c_volume2[memory_idx].x - c_volume1[memory_idx].y * c_volume2[memory_idx].y;
        im = c_volume1[memory_idx].x * c_volume2[memory_idx].y + c_volume1[memory_idx].y * c_volume2[memory_idx].x;
    }
    // printf("%f %f \n", c_volume1[memory_idx].x, c_volume2[memory_idx].y);
    c_output[memory_idx].x = re*2.0/full_vol;
    c_output[memory_idx].y = im*2.0/full_vol;

}

void cpu_VolumeConv(REAL *d_volume1,  REAL *d_volume2,  REAL *d_output, int batch_size, int volume_size, bool conjugate){
    cufftHandle plan_fwd, plan_bwd;
    cufftComplex *c_volume1, *c_volume2, *c_output;
    int reduced_volume_size = (volume_size/2 + 1);
    
    gpuErrchk(cudaMalloc((void**)&c_volume1, sizeof(cufftComplex)*batch_size*volume_size*volume_size*reduced_volume_size));
    gpuErrchk(cudaMalloc((void**)&c_volume2, sizeof(cufftComplex)*batch_size*volume_size*volume_size*reduced_volume_size));
    gpuErrchk(cudaMalloc((void**)&c_output, sizeof(cufftComplex)*batch_size*volume_size*volume_size*volume_size));
    // printf("Finished malloc\n");
    int dimensions_real[] = {volume_size, volume_size, volume_size};
    int dimensions_complex[] = {volume_size, volume_size, reduced_volume_size};
    int batch_volume_real = volume_size*volume_size*volume_size;
    int batch_volume_complex = volume_size*volume_size*reduced_volume_size;
    int inembed[] = {volume_size, volume_size, volume_size};
    int onembed[] = {volume_size, volume_size, reduced_volume_size};
    // printf("Started plan\n");
    gpuFFTErrchk(cufftPlanMany(&plan_fwd, 3, dimensions_real, 
                    inembed, 1, batch_volume_real, 
                    onembed, 1, batch_volume_complex,
                    CUFFT_R2C, batch_size));

    gpuFFTErrchk(cufftPlanMany(  &plan_bwd, 3, dimensions_real, 
                    onembed, 1, batch_volume_complex, 
                    inembed, 1, batch_volume_real,
                    CUFFT_C2R, batch_size));
    // printf("Started fft\n");
    gpuFFTErrchk(cufftExecR2C(plan_fwd, d_volume1, c_volume1));
    gpuFFTErrchk(cufftExecR2C(plan_fwd, d_volume2, c_volume2));
    // printf("Finished fft\n");

    dim3 dim_special(batch_size, batch_volume_complex/WARP_SIZE + 1);
	conjMul<<<dim_special, WARP_SIZE>>>(c_volume1, c_volume2, c_output, batch_size, volume_size, conjugate);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // printf("Finished mul\n");
    gpuFFTErrchk(cufftExecC2R(plan_bwd, c_output, d_output));
    // printf("Finished invfft\n");
    
    gpuFFTErrchk(cufftDestroy(plan_fwd));
    gpuFFTErrchk(cufftDestroy(plan_bwd));
    gpuErrchk(cudaFree(c_volume1));
    gpuErrchk(cudaFree(c_volume2));
    gpuErrchk(cudaFree(c_output));
    // printf("Finished free\n");
}


