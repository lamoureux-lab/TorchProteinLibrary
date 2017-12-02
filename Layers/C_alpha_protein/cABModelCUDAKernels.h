#include <THC/THC.h>

void cpu_computeCoordinates(float *d_alpha, float *d_beta,  // angles
							float *d_atoms,                 //atomic coords: atoms x 3
							float *d_A,                     //A-matrixes
							int L);

void cpu_computeDerivatives(float *d_alpha, float *d_beta,      // angles
							float *d_dRdAlpha, float *d_dRdBeta,//storage atoms x angles x 3
							float *d_A,                         //A-matrixes
							int L);

void cpu_backwardFromCoords(float *d_dalpha, float *d_dbeta, // angles gradients arrays
							float *d_dr,                    // coordinates gradients: 3 x atoms
							float *d_dRdAlpha, float *d_dRdBeta, //dr_j/dq_k derivatives, size=atoms x angles x 3
							int L                          //number of angles
							);

void cpu_computeBMatrix( 	float *d_dalpha, float *d_dbeta,
							float *d_coords,
							float *d_B_bend,	// L x L + 1 x 3 matrix
							float *d_B_rot,	// L x L + 1 x 3 matrix
							int L);