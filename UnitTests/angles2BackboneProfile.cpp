#include <TH.h>
#include <THC.h>
#include <iostream>
#include <cBackboneProteinCUDAKernels.h>

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
    int max_length = 650;
    int batch_size = 64;
    CUDA_REAL_TENSOR_VAR *input_angles = CUDA_REAL_TENSOR(newWithSize3d)(state, batch_size, 2, max_length);
    CUDA_REAL_TENSOR_VAR *output_coords = CUDA_REAL_TENSOR(newWithSize2d)(state, batch_size, 3*max_length*3);
    THCudaIntTensor *length = THCudaIntTensor_newWithSize1d(state, batch_size);
    CUDA_REAL_TENSOR_VAR *A = CUDA_REAL_TENSOR(newWithSize2d)(state, batch_size, 3*max_length*16);

    CUDA_REAL_TENSOR_VAR *gradInput = CUDA_REAL_TENSOR(newWithSize3d)(state, batch_size, 2, max_length);
    CUDA_REAL_TENSOR_VAR *gradOutput = CUDA_REAL_TENSOR(newWithSize2d)(state, batch_size, 3*max_length*3);
    CUDA_REAL_TENSOR_VAR *dr_dangle = CUDA_REAL_TENSOR(newWithSize3d)(state, batch_size, 2, 3*max_length*max_length*3);

    for(int i=0; i<batch_size; i++)
        THCudaIntTensor_set1d(state, length, i, max_length);

    //Forward pass
    cudaEventRecord(startFwd);
    cpu_computeCoordinatesBackbone(	CUDA_REAL_TENSOR(data)(state, input_angles), 
							CUDA_REAL_TENSOR(data)(state, output_coords), 
							CUDA_REAL_TENSOR(data)(state, A), 
							THCudaIntTensor_data(state, length),
                            input_angles->size[0],
                            input_angles->size[2]);
    cudaEventRecord(stopFwd);
    cudaEventSynchronize(stopFwd);

    
    //Backward pass
    cudaEventRecord(startBwd1);
    cpu_computeDerivativesBackbone( CUDA_REAL_TENSOR(data)(state, input_angles),
                                    CUDA_REAL_TENSOR(data)(state, dr_dangle),
                                    CUDA_REAL_TENSOR(data)(state, A),
                                    THCudaIntTensor_data(state, length),
                                    input_angles->size[0],
                                    input_angles->size[2],
                                    true);
    cudaEventRecord(stopBwd1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaEventSynchronize(stopBwd1);

    cudaEventRecord(startBwd2);
    cpu_backwardFromCoordsBackbone( CUDA_REAL_TENSOR(data)(state, gradInput),
                            CUDA_REAL_TENSOR(data)(state, gradOutput),
                            CUDA_REAL_TENSOR(data)(state, dr_dangle),
                            THCudaIntTensor_data(state, length),
                            input_angles->size[0],
                            input_angles->size[2]);
    cudaEventRecord(stopBwd2);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaEventSynchronize(stopBwd2);


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startFwd, stopFwd);
    std::cout<<"Forward, ms = "<<milliseconds<<std::endl;
    cudaEventElapsedTime(&milliseconds, startBwd1, stopBwd1);
    std::cout<<"Backward 1, ms = "<<milliseconds<<std::endl;
    cudaEventElapsedTime(&milliseconds, startBwd2, stopBwd2);
    std::cout<<"Backward 2, ms = "<<milliseconds<<std::endl;


    THCudaShutdown(state);
	free(state);
	
	return 0;
}