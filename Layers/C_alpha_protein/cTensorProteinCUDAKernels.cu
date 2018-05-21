#include "cTensorProteinCUDAKernels.h"
// #include "cMathCUDAKernels.h"
#include "cMathCUDAKernels.cu"


__global__ void backwardFromCoordinates(double *angles, double *dr, double *dR_dangle, int *length, int angles_stride){
	int batch_idx = blockIdx.x;
	int batch_size = blockDim.x;
	int atom_idx = threadIdx.x;
	int num_angles = length[batch_idx];
	int num_atoms = num_angles + 1;
	int atoms_stride = angles_stride + 1;
	
	
	double *dRdAlpha = dR_dangle + 2*batch_idx*batch_size*atoms_stride*atoms_stride*3;
	double *dRdBeta = dR_dangle + (2*batch_idx+1)*batch_size*atoms_stride*atoms_stride*3;
	double *d_dalpha = angles + (2*batch_idx)*angles_stride;
	double *d_dbeta = angles + (2*batch_idx+1)*angles_stride;
	double *d_dr = dr + batch_idx*atoms_stride*3;
	for(int j=atom_idx+1; j<num_atoms; j++){
		int index_upper = atom_idx*num_atoms+j;
		d_dalpha[atom_idx] += vec3Mul(d_dr+3*j, dRdAlpha + 3*index_upper);
		d_dbeta[atom_idx] += vec3Mul(d_dr+3*j, dRdBeta + 3*index_upper);
	}
}


__global__ void computeCoordinatesDihedral( double *angles,	double *atoms, double *A, int *length, int angles_stride){
	uint k = (blockIdx.x * blockDim.x) + threadIdx.x;
	double *d_atoms = atoms + k*(angles_stride)*3;
	double *d_phi = angles + 2*k*angles_stride;
	double *d_psi = angles + (2*k+1)*angles_stride;
	double *d_A = A + k*angles_stride*16;
	int num_angles = length[k];

	
	double B[16];
	double origin[3]; origin[0]=0.0; origin[1]=0.0; origin[2]=0.0;
	getRotationMatrixCalpha(d_A, d_phi[0], 0.0, true);
	mat44Vec3Mul(d_A, origin, d_atoms);
	
	for(int i=1; i<num_angles; i++){
		getRotationMatrixCalpha(B, d_phi[i], d_psi[i-1], false);
		mat44Mul(d_A+16*(i-1), B, d_A+16*i);
		mat44Vec3Mul(d_A+16*i, origin, d_atoms + 3*(i));
	}
}

__global__ void computeGradientsOptimizedDihedral( double *angles, double *dR_dangle, double *A, int *length, int angles_stride){
							
	uint k = (blockIdx.x * blockDim.x) + threadIdx.x;
	int num_angles = length[blockIdx.x];
	int num_atoms = num_angles + 1;
	int atoms_stride = angles_stride+1;
	double *d_alpha = angles + 2*k*angles_stride;
	double *d_beta = angles + (2*k+1)*angles_stride;
	double *d_dRdAlpha = dR_dangle + blockIdx.x * (atoms_stride*angles_stride*3) + threadIdx.x*atoms_stride*3;
	double *d_dRdBeta = dR_dangle + blockIdx.x * (atoms_stride*angles_stride*3) + threadIdx.x*atoms_stride*3;
	double *d_A = A + blockIdx.x * angles_stride * 16;
	double r_0[3];setVec3(r_0, 0, 0, 0);
	double dBdAlpha[16], dBdBeta[16], leftPartAlpha[16], leftPartBeta[16], rightPart[16];
	double tmp[16], B[16];
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
	for(int j=k+1; j<num_atoms; j++){
		int index_upper = k*atoms_stride+j;
		mat44Mul(leftPartAlpha, rightPart, tmp);
		mat44Vec3Mul(tmp, r_0, d_dRdAlpha + 3*index_upper);
		mat44Mul(leftPartBeta, rightPart, tmp);
		mat44Vec3Mul(tmp, r_0, d_dRdBeta + 3*index_upper);
		getRotationMatrixCalpha(B, d_alpha[j], d_beta[j], false);
		mat44Mul(rightPart, B, rightPart);
	}
}

void cpu_computeCoordinatesDihedral(double *angles,double *atoms,double *A,int *length,int batch_size,int angles_stride){
	computeCoordinatesDihedral<<<batch_size,1>>>(angles, atoms, A, length, angles_stride);
}

void cpu_computeDerivativesDihedral(double *angles,double *dR_dangle,double *A,int *length, int batch_size,int angles_stride){
	computeGradientsOptimizedDihedral<<<angles_stride, batch_size>>>(angles, dR_dangle, A, length, angles_stride);
}

void cpu_backwardFromCoords(double *angles, double *dr, double *dR_dangle, int *length, int batch_size, int angles_stride){
	backwardFromCoordinates<<<angles_stride, batch_size>>>(angles, dr, dR_dangle, length, angles_stride);
}
