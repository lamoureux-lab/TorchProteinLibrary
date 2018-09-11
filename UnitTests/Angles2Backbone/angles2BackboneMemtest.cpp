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


int main(int argc, char** argv)
{
    
    int max_length = 650;
    int batch_size = 64;
    // int batch_size = 1;
    // int max_length = 1;
    float *input_angles, *output_coords, *A;
    float *gradInput, *gradOutput, *dr_dangle;
    int *length, *cpu_length;
    cudaMalloc((void**)&input_angles, sizeof(float)*batch_size*2*max_length);
    cudaMalloc((void**)&output_coords, sizeof(float)*batch_size*3*max_length*3);
    cudaMalloc((void**)&A, sizeof(float)*batch_size*3*max_length*16);

    cudaMalloc((void**)&gradInput, sizeof(float)*batch_size*2*max_length);
    cudaMalloc((void**)&gradOutput, sizeof(float)*batch_size*3*max_length*3);
    cudaMalloc((void**)&dr_dangle, sizeof(float)*batch_size*2*3*max_length*max_length*3);

    cudaMalloc((void**)&length, sizeof(int)*batch_size);
    cpu_length = new int[batch_size];
    for(int i=0;i<batch_size;i++)cpu_length[i]=max_length;
    cudaMemcpy(length, cpu_length, sizeof(int)*batch_size, cudaMemcpyHostToDevice);
    
    size_t total_mem_fwd = sizeof(float)*batch_size*max_length*(2 + 3*3 + 3*16) + sizeof(int)*batch_size;
    size_t total_mem_bwd = sizeof(float)*batch_size*max_length*(2 + 3*3 + 2*3*3*max_length);
    std::cout<<"Forward pass mem (mb): "<<double(total_mem_fwd)/(1024. * 1024.)<<"\n";
    std::cout<<"Backward pass mem (mb): "<<double(total_mem_bwd)/(1024. * 1024.)<<"\n";
    
    //Forward pass
    cpu_computeCoordinatesBackbone(	input_angles, output_coords, A,length, batch_size, max_length);
    
    //Backward pass
    cpu_computeDerivativesBackbone( input_angles, dr_dangle, A, length, batch_size, max_length);
    cpu_backwardFromCoordsBackbone( gradInput, gradOutput, dr_dangle, length, batch_size, max_length, true);
    
    cudaFree(input_angles);
    cudaFree(output_coords);
    cudaFree(A);
    cudaFree(gradInput);
    cudaFree(gradOutput);
    cudaFree(dr_dangle);
    cudaFree(length);
    delete [] cpu_length;

	return 0;
}