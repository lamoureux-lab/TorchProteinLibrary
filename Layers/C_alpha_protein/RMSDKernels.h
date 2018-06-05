#include <THC/THC.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void cpu_correlationMatrix(     double *d_coords1,  //input: coordinates 1
                                double *d_coords2,  //input: coordinates 2
                                double *T,  //output: T-correlation matrix
                                int *num_atoms, int batch_size, int coords_stride);            //param: number of angles

void cpu_computeR2( double *d_coordinates, int num_atoms, double *R2);

void cpu_transformCoordinates( double *d_coordinates_src, //input: coordinates to transform
                                double *d_coordinates_dst,   //output: transformed coordinates
                                double *d_matrix,            //input: transformation matrix
                                int batch_size, int coords_stride);


THCudaDoubleTensor* toGPU(THCState*state, THDoubleTensor *T);
THDoubleTensor* fromGPU(THCState*state, THCudaDoubleTensor *T);
void toGPUTensor(THCState*state, double *cpu_T, THCudaDoubleTensor *gpu_T);
void toCPUTensor(THCState*state, THDoubleTensor *cpu_T, THCudaDoubleTensor *gpu_T);
