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

void cpu_computeCentroid(   float *d_coords,  //input: coordinates 
                            float *d_centroid,  //output: centroid 
                            int L );            //param: number of angles

void cpu_centerCoords(      float *d_coords_src,  //input: initial coordinates 
                            float *d_centroid,  //input: centroid 
                            float *d_coords_dst,  //output: centered coordinates 
                            int L );            //param: number of angles

void cpu_correlationMatrix(     float *d_coords1,  //input: coordinates 1
                                float *d_coords2,  //input: coordinates 2
                                double *T,  //output: T-correlation matrix
                                int L );            //param: number of angles

float cpu_constructRotMatrix(    float *d_eigenvectors,  //input: T-eigenvectors
                                float *d_eigenvalues,   //input: T-eigenvalues
                                float *U);              //output: rotation matrix

void cpu_computeR2( float *d_coordinates, int L, double *R2);

void cpu_transformCoordinates( float *d_coordinates_src, //input: coordinates to transform
                                float *d_coordinates_dst,   //output: transformed coordinates
                                float *d_matrix,            //input: transformation matrix
                                int L);                     //param: number of angles

void cpu_mulAndSub( float *d_coordinates_dst,
                    float *d_coordinates1,
                    float *d_coordinates2,
                    float mult,
                    int L);

THCudaTensor* toGPU(THCState*state, THFloatTensor *T);
THFloatTensor* fromGPU(THCState*state, THCudaTensor *T);
THDoubleTensor* fromGPUDouble(THCState*state, THCudaDoubleTensor *T);
void toGPUTensor(THCState*state, float *cpu_T, THCudaTensor *gpu_T);
void toCPUTensor(THCState*state, THFloatTensor *cpu_T, THCudaTensor *gpu_T);
