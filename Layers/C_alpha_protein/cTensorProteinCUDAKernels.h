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


void cpu_backwardFromCoords(double *angles, 
							double *dr, 
							double *dR_dangle, 
							int *length, 
							int batch_size, 
							int angles_stride);

void cpu_computeCoordinatesDihedral(double *angles,  
									double *atoms,   
									double *A,       
									int *length,
									int batch_size,
									int angles_stride);

void cpu_computeDerivativesDihedral(double *angles,
									double *dR_dangle,
									double *A,
									int *length,
									int batch_size,
									int angles_stride);