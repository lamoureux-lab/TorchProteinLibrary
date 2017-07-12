#include "cTensorProteinCUDAKernels.h"
// #include "cMathCUDAKernels.h"
#include "cMathCUDAKernels.cu"

// #define KAPPA1 (3.14159 - 1.9391)
// #define KAPPA2 (3.14159 - 2.0355)
// #define KAPPA3 (3.14159 - 2.1186)
// #define KAPPA1 (3.14159 - 1.9391)


__global__ void computeCoordinates( float *d_phi, float *d_psi, // angles arrays
									float *d_atoms_calpha,
									float *d_atoms_c,
									float *d_atoms_n,
									float *d_A,
									int L){                          //number of angles
									
	float A[16],B[16],C[16];

	setVec3(d_atoms_n, 0, 0, 0);
	for(int i=0; i<L; i++){
		getRotationMatrixDihedral(A, d_phi[i], KAPPA1, R_N_CALPHA);
		getRotationMatrixDihedral(B, d_psi[i], KAPPA2, R_CALPHA_C);
		getRotationMatrixDihedral(C, OMEGACIS, KAPPA3, R_C_N);
		
		if(i>0){
			mat44Mul(d_A+16*(3*(i-1)+2), A, d_A+16*(3*i));
		}else{
			getRotationMatrixDihedral(d_A, d_phi[i], KAPPA1, R_N_CALPHA);
		}
		mat44Mul(d_A+16*(3*i), B, d_A+16*(3*i+1));
		mat44Mul(d_A+16*(3*i+1), C, d_A+16*(3*i+2));
		
		mat44Vec3Mul(d_A+16*(3*i), d_atoms_n, d_atoms_calpha + 3*(i));
		mat44Vec3Mul(d_A+16*(3*i+1), d_atoms_n, d_atoms_c + 3*(i));
		if(i<(L-1))
			mat44Vec3Mul(d_A+16*(3*i+2), d_atoms_n, d_atoms_n + 3*(i+1));
	}
}

__global__ void computeCoordinatesCalpha( 	float *d_phi, float *d_psi, // angles arrays
											float *d_atoms,                 //atomic coords size = atoms x 3
											float *d_A,                     //A-matrixes, saved for backward pass
											int L){
	setVec3(d_atoms, 0, 0, 0);
	float B[16];
	getRotationMatrixCalpha(d_A, d_phi[0], d_psi[0]);
	for(int i=0; i<L; i++){
		getRotationMatrixCalpha(B, d_phi[i], d_psi[i]);
		if(i>0){            
			mat44Mul(d_A+16*(i-1), B, d_A+16*i);
		}
		mat44Vec3Mul(d_A+16*i, d_atoms, d_atoms + 3*(i+1));
	}
}

__global__ void computeGradientsOptimized(
								float *d_alpha, float *d_beta, // angles arrays
								float *d_dRdAlpha, float *d_dRdBeta, //dr_j/dq_k derivatives, size=atoms x angles x 3
								float *d_A,                     //A-matrixes, computed during forward
								int L){                          //number of angles
								
	uint k = (blockIdx.x * blockDim.x) + threadIdx.x;
	int atoms_size = L+1;
	float r_0[3];setVec3(r_0, 0, 0, 0);
	float dBdAlpha[16], dBdBeta[16], leftPartAlpha[16], leftPartBeta[16], rightPart[16];
	float tmp[16], B[16];
	getRotationMatrixCalphaDPhi(dBdAlpha, d_alpha[k], d_beta[k]);
	getRotationMatrixCalphaDPsi(dBdBeta, d_alpha[k], d_beta[k]);
	if(k>0){
		mat44Mul(d_A + 16*(k-1), dBdAlpha, leftPartAlpha);
		mat44Mul(d_A + 16*(k-1), dBdBeta, leftPartBeta);
	}else{
		memcpy(leftPartAlpha, dBdAlpha, 16*sizeof(float));
		memcpy(leftPartBeta, dBdBeta, 16*sizeof(float));
	}
	getIdentityMatrix44(rightPart);
	for(int j=k+1; j<atoms_size; j++){
		int index_upper = k*atoms_size+j;
		mat44Mul(leftPartAlpha, rightPart, tmp);
		mat44Vec3Mul(tmp, r_0, d_dRdAlpha + 3*index_upper);
		mat44Mul(leftPartBeta, rightPart, tmp);
		mat44Vec3Mul(tmp, r_0, d_dRdBeta + 3*index_upper);
		getRotationMatrixCalpha(B, d_alpha[j], d_beta[j]);
		mat44Mul(rightPart, B, rightPart);
	}
}

// __global__ void computeGradientsOptimized(
// 								float *d_phi, float *d_psi, // angles arrays
// 								float *d_dr_calpha_dphi, float *d_dr_calpha_dpsi, //dr_j/dq_k derivatives, size=atoms x angles x 3
// 								float *d_dr_c_dphi, float *d_dr_c_dpsi, //dr_j/dq_k derivatives, size=atoms x angles x 3
// 								float *d_dr_n_dphi, float *d_dr_n_dpsi, //dr_j/dq_k derivatives, size=atoms x angles x 3
// 								float *d_A,                     //A-matrixes, computed during forward
// 								int L){                          //number of angles
								
// 	uint k = (blockIdx.x * blockDim.x) + threadIdx.x;
// 	int atoms_size = L+1;
// 	float r_0[3];setVec3(r_0, 0, 0, 0);
// 	float dB_calpha_dphi[16], dB_calpha_dpsi[16], dB_c_dphi[16], dB_c_dpsi[16], dB_n_dphi[16], dB_n_dpsi[16];
// 	float leftPart_calpha_Phi[16], leftPart_calpha_Psi[16], rightPart_calpha[16];
// 	float leftPart_c_Phi[16], leftPart_c_Psi[16], rightPart_c[16];
// 	float leftPart_n_Phi[16], leftPart_n_Psi[16], rightPart_n[16];
// 	float tmp[16], B[16];
// 	getRotationMatrixDAlpha(dBdphi, d_alpha[k], d_beta[k], R);
// 	getRotationMatrixDBeta(dBdpsi, d_alpha[k], d_beta[k], R);
	
// 	getRotationMatrixDihedralDPsi(dBdphi, d_phi[k], KAPPA1, R_N_CALPHA);
// 	getRotationMatrixDihedralDPsi(dBdpsi, d_psi[k], KAPPA2, R_CALPHA_C);
// 	getRotationMatrixDihedralDPsi(C, OMEGACIS, KAPPA3, R_C_N);

// 	if(k>0){
// 		mat44Mul(d_A + 16*(k-1), dBdphi, leftPartPhi);
// 		mat44Mul(d_A + 16*(k-1), dBdpsi, leftPartPsi);
// 	}else{
// 		memcpy(leftPartPhi, dBdphi, 16*sizeof(float));
// 		memcpy(leftPartPsi, dBdpsi, 16*sizeof(float));
// 	}
// 	getIdentityMatrix44(rightPart);
// 	for(int j=k+1; j<atoms_size; j++){
// 		int index_upper = k*atoms_size+j;
// 		mat44Mul(leftPartPhi, rightPart, tmp);
// 		mat44Vec3Mul(tmp, r_0, d_drdphi + 3*index_upper);
// 		mat44Mul(leftPartPsi, rightPart, tmp);
// 		mat44Vec3Mul(tmp, r_0, d_drdpsi + 3*index_upper);
// 		getRotationMatrix(B, d_alpha[j], d_beta[j], R);
// 		mat44Mul(rightPart, B, rightPart);
// 	}
// }

__global__ void backwardFromCoordinates(
								float *d_dalpha, float *d_dbeta, // angles gradients arrays
								float *d_dr,                    // coordinates gradients: 3 x atoms
								float *dRdAlpha, float *dRdBeta, //dr_j/dq_k derivatives, size=atoms x angles x 3
								int L                          //number of angles
								){
	int angles_size = L;
	int atoms_size = L+1;
	uint k = (blockIdx.x * blockDim.x) + threadIdx.x;
	// d_dalpha[k]=0.0;
	// d_dbeta[k]=0.0;
	for(int j=k+1; j<atoms_size; j++){
		int index_upper = k*atoms_size+j;
		d_dalpha[k] += vec3Mul(d_dr+3*j, dRdAlpha + 3*index_upper);
		d_dbeta[k] += vec3Mul(d_dr+3*j, dRdBeta + 3*index_upper);
	}
}


void cpu_computeCoordinates(float *d_phi, float *d_psi, // angles arrays
							float *d_atoms_calpha,
							float *d_atoms_c,
							float *d_atoms_n,               
							float *d_A,                     //A-matrixes
							int L){                //params
	computeCoordinates<<<1,1>>>(d_phi, d_psi, d_atoms_calpha, d_atoms_c, d_atoms_n, d_A, L);
}

void cpu_computeCoordinatesCalpha( 	float *d_phi, float *d_psi, // angles arrays
											float *d_atoms,                 //atomic coords size = atoms x 3
											float *d_A,                     //A-matrixes, saved for backward pass
											int L){
	computeCoordinatesCalpha<<<1,1>>>(d_phi, d_psi, d_atoms, d_A, L);
}

void cpu_computeDerivativesCalpha(float *d_phi, float *d_psi,      // angles
							float *d_drdphi, float *d_drdpsi,//storage atoms x angles x 3
							float *d_A,                         //A-matrixes
							int L){                    //params    
	computeGradientsOptimized<<<1,L>>>(d_phi, d_psi, d_drdphi, d_drdpsi, d_A, L);
}

void cpu_backwardFromCoordsCalpha(float *d_dalpha, float *d_dbeta, // angles gradients arrays
							float *d_dr,                    // coordinates gradients: 3 x atoms
							float *d_dRdAlpha, float *d_dRdBeta, //dr_j/dq_k derivatives, size=atoms x angles x 3
							int L                          //number of angles
							){                   
	backwardFromCoordinates<<<1,L>>>(d_dalpha, d_dbeta, d_dr, d_dRdAlpha, d_dRdBeta, L);
}
