#include "cTensorProteinCUDAKernels.h"
// #include "cMathCUDAKernels.h"
#include "cMathCUDAKernels.cu"


__global__ void computeCoordinatesDihedral( double *angles,	double *atoms, double *A, int *length, int angles_stride){
	uint batch_idx = threadIdx.x;

	double *d_atoms = atoms + batch_idx*(angles_stride)*3;
	double *d_phi = angles + 2*batch_idx*angles_stride;
	double *d_psi = angles + (2*batch_idx+1)*angles_stride;
	double *d_A = A + batch_idx*angles_stride*16;
	int num_angles = length[batch_idx];

	
	double B[16];
	double origin[3];setVec3(origin, 0, 0, 0);
	getRotationMatrixCalpha(d_A, d_phi[0], 0.0, true);
	mat44Vec3Mul(d_A, origin, d_atoms);
	
	for(int i=1; i<num_angles; i++){
		getRotationMatrixCalpha(B, d_phi[i], d_psi[i-1], false);
		mat44Mul(d_A+16*(i-1), B, d_A+16*i);
		mat44Vec3Mul(d_A+16*i, origin, d_atoms + 3*(i));
	}
}

__global__ void computeGradientsOptimizedDihedral( double *angles, double *dR_dangle, double *A, int *length, int angles_stride){
	// uint batch_size = blockDim.x;	
	// uint batch_idx = blockIdx.x;
	// uint atom_i_idx = threadIdx.x;
	// uint angle_k_idx = threadIdx.y;
	uint batch_size = blockDim.x;
	uint batch_idx = blockIdx.x;
	uint atom_i_idx = blockIdx.y;
	uint angle_k_idx = blockIdx.z;
	
	int num_angles = length[batch_idx];
	int num_atoms = num_angles;
	int atoms_stride = angles_stride;
	double *d_A = A + batch_idx * angles_stride * 16;

	double *d_phi = angles + 2*batch_idx*angles_stride;
	double *d_psi = angles + (2*batch_idx+1)*angles_stride;

	double *dR_dPhi = dR_dangle + 2*batch_idx * (atoms_stride*angles_stride*3) + angle_k_idx*atoms_stride*3 + atom_i_idx*3;
	double *dR_dPsi = dR_dangle + (2*batch_idx+1) * (atoms_stride*angles_stride*3) + angle_k_idx*atoms_stride*3 + atom_i_idx*3;

	if(atom_i_idx<angle_k_idx){
		setVec3(dR_dPsi, 0, 0, 0);
		setVec3(dR_dPhi, 0, 0, 0);
		return;
	}
	
	double origin[3];setVec3(origin, 0, 0, 0);
	double dBk_dPhik[16], dBk1_dPsik[16];	
	double Ak1_inv[16], Ak_inv[16];
	double tmp1[16], tmp2[16], tmp3[16];

	if(angle_k_idx>0){
		getRotationMatrixCalphaDPhi(dBk_dPhik, d_phi[angle_k_idx], d_psi[angle_k_idx-1], false);
	}else{
		getRotationMatrixCalphaDPhi(dBk_dPhik, d_phi[angle_k_idx], 0.0, true);
	}
	getRotationMatrixCalphaDPsi(dBk1_dPsik, d_phi[angle_k_idx+1], d_psi[angle_k_idx]);

	// double L[16], Linv[16];
	// invertMat44(Linv, d_A + angle_k_idx*16);
	// mat44Mul(d_A + angle_k_idx*16, Linv, L);
	// printf("%f %f %f %f\n", L[0], L[1], L[2], L[3]);
	// printf("%f %f %f %f\n", L[4], L[5], L[6], L[7]);
	// printf("%f %f %f %f\n", L[8], L[9], L[10], L[11]);
	// printf("%f %f %f %f\n", L[12], L[13], L[14], L[15]);
	
	//dA_i / dphi_k
	invertMat44(Ak_inv, d_A + angle_k_idx*16);
	
	if(angle_k_idx>0)
		mat44Mul(d_A + (angle_k_idx-1)*16, dBk_dPhik, tmp1);
	else
		setMat44(tmp1, dBk_dPhik);
	mat44Mul(Ak_inv, d_A + atom_i_idx*16, tmp2);
	mat44Mul(tmp1, tmp2, tmp3);
	mat44Vec3Mul(tmp3, origin, dR_dPhi);
	
	//dA_i / dpsi_k
	if(angle_k_idx == atom_i_idx){
		setVec3(dR_dPsi, 0, 0, 0);
		return;
	}
	invertMat44(Ak1_inv, d_A + (angle_k_idx + 1)*16);

	mat44Mul(d_A + angle_k_idx*16, dBk1_dPsik, tmp1);
	mat44Mul(Ak1_inv, d_A + atom_i_idx*16, tmp2);
	mat44Mul(tmp1, tmp2, tmp3);
	mat44Vec3Mul(tmp3, origin, dR_dPsi);
	
	
}

__global__ void backwardFromCoordinates(double *angles, double *dr, double *dR_dangle, int *length, int angles_stride){
	int batch_idx = blockIdx.x;
	int batch_size = blockDim.x;
	int angle_idx = threadIdx.x;
	int num_angles = length[batch_idx];
	int num_atoms = num_angles;
	int atoms_stride = angles_stride;
	
	double *d_phi = angles + 2*batch_idx*angles_stride + angle_idx;
	double *d_psi = angles + (2*batch_idx+1)*angles_stride + angle_idx;
	
	double *dR_dPhi = dR_dangle + 2*batch_idx * (atoms_stride*angles_stride*3) + angle_idx*atoms_stride*3;
	double *dR_dPsi = dR_dangle + (2*batch_idx+1) * (atoms_stride*angles_stride*3) + angle_idx*atoms_stride*3;

	double *d_dr = dr + batch_idx*atoms_stride*3;

	for(int j=angle_idx; j<num_atoms; j++){
		
		(*d_phi) += vec3Mul(d_dr+3*j, dR_dPhi + 3*j);
		(*d_psi) += vec3Mul(d_dr+3*j, dR_dPsi + 3*j);
	}
}


void cpu_computeCoordinatesDihedral(double *angles, double *atoms, double *A, int *length, int batch_size, int angles_stride){
	computeCoordinatesDihedral<<<1,batch_size>>>(angles, atoms, A, length, angles_stride);
}

void cpu_computeDerivativesDihedral(double *angles, double *dR_dangle, double *A, int *length, int batch_size, int angles_stride){
	// dim3 angles_dim(angles_stride, angles_stride, 1);
	// computeGradientsOptimizedDihedral<<<batch_size, angles_dim>>>(angles, dR_dangle, A, length, angles_stride);
	dim3 batch_angles_dim(batch_size, angles_stride, angles_stride);
	computeGradientsOptimizedDihedral<<<batch_angles_dim, 1>>>(angles, dR_dangle, A, length, angles_stride);
}

void cpu_backwardFromCoords(double *angles, double *dr, double *dR_dangle, int *length, int batch_size, int angles_stride){
	backwardFromCoordinates<<<batch_size, angles_stride>>>(angles, dr, dR_dangle, length, angles_stride);
}
