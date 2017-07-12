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



void cpu_computeCoordinates(float *d_phi, float *d_psi, // angles arrays
							float *d_atoms_calpha,
							float *d_atoms_c,
							float *d_atoms_n,               
							float *d_A,                     //A-matrixes
							int L);                //params


void cpu_computeCoordinatesCalpha( 	float *d_phi, float *d_psi, // angles arrays
											float *d_atoms,                 //atomic coords size = atoms x 3
											float *d_A,                     //A-matrixes, saved for backward pass
											int L);

void cpu_computeDerivativesCalpha(float *d_phi, float *d_psi,      // angles
                            float *d_drdphi, float *d_drdpsi,    //storage atoms x angles x 3
                            float *d_A,                         //A-matrixes
                            int L);                    //params    

void cpu_backwardFromCoordsCalpha(float *d_dalpha, float *d_dbeta, // angles gradients arrays
							float *d_dr,                    // coordinates gradients: 3 x atoms
							float *d_dRdAlpha, float *d_dRdBeta, //dr_j/dq_k derivatives, size=atoms x angles x 3
							int L                          //number of angles
							);                   

