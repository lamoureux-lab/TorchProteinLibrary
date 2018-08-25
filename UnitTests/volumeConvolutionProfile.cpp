#include <TH.h>
#include <THC.h>
#include <iostream>
#include <VolumeConv.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define CUDA_REAL_TENSOR_VAR THCudaTensor
#define CUDA_REAL_TENSOR(X) THCudaTensor_##X
                            
int main(int argc, char** argv)
{
    cudaEvent_t startFwd, stopFwd;
    cudaEvent_t startBwd1, stopBwd1, startBwd2, stopBwd2;
    cudaEventCreate(&startFwd);
    cudaEventCreate(&stopFwd);
    cudaEventCreate(&startBwd1);cudaEventCreate(&startBwd2);
    cudaEventCreate(&stopBwd1);cudaEventCreate(&stopBwd2);



    THCState* state = (THCState*) malloc(sizeof(THCState));    
	memset(state, 0, sizeof(THCState));
	THCudaInit(state); 
    
    int batch_size = 4;
    int vol_size = 256;
    
    CUDA_REAL_TENSOR_VAR *inputV1 = CUDA_REAL_TENSOR(newWithSize4d)(state, batch_size, vol_size, vol_size, vol_size);
    CUDA_REAL_TENSOR_VAR *inputV2 = CUDA_REAL_TENSOR(newWithSize4d)(state, batch_size, vol_size, vol_size, vol_size);
    CUDA_REAL_TENSOR_VAR *outputV = CUDA_REAL_TENSOR(newWithSize4d)(state, batch_size, vol_size, vol_size, vol_size);
    
    //Forward pass
    cudaEventRecord(startFwd);
    cpu_VolumeConv(	CUDA_REAL_TENSOR(data)(state, inputV1), 
					CUDA_REAL_TENSOR(data)(state, inputV2), 
					CUDA_REAL_TENSOR(data)(state, outputV), 
                    batch_size, vol_size);

    cudaEventRecord(stopFwd);
    cudaEventSynchronize(stopFwd);

    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startFwd, stopFwd);
    std::cout<<"Forward, ms = "<<milliseconds<<std::endl;
    

    CUDA_REAL_TENSOR(free)(state, inputV1);
    CUDA_REAL_TENSOR(free)(state, inputV2);
    CUDA_REAL_TENSOR(free)(state, outputV);
    THCudaShutdown(state);
	free(state);
	
	return 0;
}