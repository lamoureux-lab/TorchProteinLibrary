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


void cpu_computeDerivatives(float *d_alpha, float *d_beta,      // angles
                            float *d_dRdAlpha, float *d_dRdBeta,    //storage atoms x angles x 3
                            float *d_A,                         //A-matrixes
                            int L, float R);                    //params    

void cpu_computeCoordinates(float *d_alpha, float *d_beta,  // angles
                            float *d_atoms,                 //atomic coords: atoms x 3
                            float *d_A,                     //A-matrixes
                            int L, float R);                //params

void cpu_backwardFromCoords(float *d_dalpha, float *d_dbeta, // angles gradients arrays
							float *d_dr,                    // coordinates gradients: 3 x atoms
							float *d_dRdAlpha, float *d_dRdBeta, //dr_j/dq_k derivatives, size=atoms x angles x 3
							int L                          //number of angles
							);                   

void cpu_computePairCoordinates(float *coords,               // coordinates
                                float *distances,           //pairwise coords: atoms x atoms x 3
                                int L,                     // num angles
								int Lmax);					// max num angles

void cpu_backwardFromPairCoordinates(	float *grad_coords,               // gradient of coordinates
                                		float *grad_distances,           // gradient of pairwise coords: atoms x atoms x 3
                               		 	int L,                     // num angles
										int Lmax);					// max num angles


void cpu_computeBasis(		    float *d_alpha, float *d_beta, // angles arrays
								float *d_axisx,                 //coordinates of the transformed x-axis size = atoms x 3
								float *d_axisy,                 //coordinates of the transformed y-axis size = atoms x 3
								float *d_axisz,                 //coordinates of the transformed z-axis size = atoms x 3
								float *d_B,                     //3x3 rotation matrixes, saved for backward pass
								int L);                         //number of angles

void cpu_computeBasisGradients(			float *d_alpha, float *d_beta, 			// angles arrays
										float *d_daxdAlpha, float *d_daxdBeta, 	//dax_j/dq_k derivatives, size=atoms x angles x 3
										float *d_daydAlpha, float *d_daydBeta, 	//day_j/dq_k derivatives, size=atoms x angles x 3
										float *d_dazdAlpha, float *d_dazdBeta, 	//daz_j/dq_k derivatives, size=atoms x angles x 3
										float *d_B,                     //3x3 rotation-matrixes, computed during forward
										int L);                          //number of angles  

void cpu_backwardFromBasis(		float *d_dalpha, float *d_dbeta, // angles gradients arrays
								float *d_dax, float *d_day, float *d_daz,      // coordinates gradients: 3 x atoms
								float *d_daxdAlpha, float *d_daxdBeta, 	//dax_j/dq_k derivatives, size=atoms x angles x 3
								float *d_daydAlpha, float *d_daydBeta, 	//day_j/dq_k derivatives, size=atoms x angles x 3
								float *d_dazdAlpha, float *d_dazdBeta, 	//daz_j/dq_k derivatives, size=atoms x angles x 3
								int L);                          //number of angles
							   
void cpu_computeVolumes(int *input_types, 		//input sequence with residue types
						float *input_coords, 	//input coordinates of atoms
						float *output_volumes,  //output volumes with densities
						int L, 					//number of angles
						int num_types,			//number of residue types
						int vol_size,			//linear size of the volumes	
						float resolution);

void cpu_computeVolumesBackward(	int *input_types, 		//input sequence with residue types
									float *input_dvolumes, 	//input gradient of densities
									float *input_coords, 	//input coordinates
									float *output_dcoords,  //output gradients of coordinates
									int L, 					//number of angles
									int num_types,			//number of residue types
									int vol_size,			//size of the volumes
									float resolution);		//volume resolution

void cpu_computeCoordinatesDihedral(float *d_alpha, float *d_beta,  // angles
									float *d_atoms,                 //atomic coords: atoms x 3
									float *d_A,                     //A-matrixes
									int L, float R);
void cpu_computeDerivativesDihedral(float *d_alpha, float *d_beta,      // angles
									float *d_dRdAlpha, float *d_dRdBeta,//storage atoms x angles x 3
									float *d_A,                         //A-matrixes
									int L, float R);                    //params    

void cpu_computeBasisDihedral(  float *d_alpha, float *d_beta, // angles arrays
								float *d_axisx,                 //coordinates of the transformed x-axis size = atoms x 3
								float *d_axisy,                 //coordinates of the transformed y-axis size = atoms x 3
								float *d_axisz,                 //coordinates of the transformed z-axis size = atoms x 3
								float *d_B,                     //3x3 rotation matrixes, saved for backward pass
								int L);                         //number of angles

void cpu_computeBasisGradientsDihedral(	float *d_alpha, float *d_beta, 			// angles arrays
										float *d_daxdAlpha, float *d_daxdBeta, 	//dax_j/dq_k derivatives, size=atoms x angles x 3
										float *d_daydAlpha, float *d_daydBeta, 	//day_j/dq_k derivatives, size=atoms x angles x 3
										float *d_dazdAlpha, float *d_dazdBeta, 	//daz_j/dq_k derivatives, size=atoms x angles x 3
										float *d_B,                     //3x3 rotation-matrixes, computed during forward
										int L);                          //number of angles