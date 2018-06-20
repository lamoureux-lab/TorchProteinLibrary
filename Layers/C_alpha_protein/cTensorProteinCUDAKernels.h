#include <THC/THC.h>
#include <cMathCUDAKernels.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void cpu_backwardFromCoords(REAL *angles, 
							REAL *dr, 
							REAL *dR_dangle, 
							int *length, 
							int batch_size, 
							int angles_stride);

void cpu_computeCoordinatesDihedral(REAL *angles,  
									REAL *atoms,   
									REAL *A,       
									int *length,
									int batch_size,
									int angles_stride);

void cpu_computeDerivativesDihedral(REAL *angles,
									REAL *dR_dangle,
									REAL *A,
									int *length,
									int batch_size,
									int angles_stride);