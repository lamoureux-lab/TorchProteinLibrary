#include "cTensorProteinCUDAKernels.h"
// #include "cMathCUDAKernels.h"
#include "cMathCUDAKernels.cu"


__global__ void computeCoordinatesDihedral( REAL *angles,	REAL *atoms, REAL *A, int *length, int angles_stride){
	uint batch_idx = threadIdx.x;

	REAL *d_atoms = atoms + batch_idx*(angles_stride)*3;
	REAL *d_phi = angles + 2*batch_idx*angles_stride;
	REAL *d_psi = angles + (2*batch_idx+1)*angles_stride;
	REAL *d_A = A + batch_idx*angles_stride*16;
	int num_angles = length[batch_idx];

	
	REAL B[16];
	REAL origin[3];setVec3(origin, 0, 0, 0);
	getRotationMatrixCalpha(d_A, d_phi[0], 0.0, true);
	mat44Vec3Mul(d_A, origin, d_atoms);
	
	for(int i=1; i<num_angles; i++){
		getRotationMatrixCalpha(B, d_phi[i], d_psi[i-1], false);
		mat44Mul(d_A+16*(i-1), B, d_A+16*i);
		mat44Vec3Mul(d_A+16*i, origin, d_atoms + 3*(i));
	}
}

__global__ void computeGradientsOptimizedDihedral( REAL *angles, REAL *dR_dangle, REAL *A, int *length, int angles_stride){
	// uint batch_size = blockDim.x;	
	// uint batch_idx = blockIdx.x;
	// uint atom_i_idx = threadIdx.x;
	// uint angle_k_idx = threadIdx.y;
	
	//Time 1
	// uint batch_size = blockDim.x;
	// uint batch_idx = blockIdx.x;
	// uint atom_i_idx = blockIdx.y;
	// uint angle_k_idx = blockIdx.z;

	//Time 2
	uint batch_size = blockDim.x;
	uint batch_idx = blockIdx.x;
	uint atom_i_idx = blockIdx.y;
	uint angle_k_idx = threadIdx.x;
	
	int num_angles = length[batch_idx];
	int num_atoms = num_angles;
	int atoms_stride = angles_stride;
	REAL *d_A = A + batch_idx * angles_stride * 16;

	REAL *d_phi = angles + 2*batch_idx*angles_stride;
	REAL *d_psi = angles + (2*batch_idx+1)*angles_stride;

	REAL *dR_dPhi = dR_dangle + 2*batch_idx * (atoms_stride*angles_stride*3) + angle_k_idx*atoms_stride*3 + atom_i_idx*3;
	REAL *dR_dPsi = dR_dangle + (2*batch_idx+1) * (atoms_stride*angles_stride*3) + angle_k_idx*atoms_stride*3 + atom_i_idx*3;

	if(atom_i_idx<angle_k_idx){
		setVec3(dR_dPsi, 0, 0, 0);
		setVec3(dR_dPhi, 0, 0, 0);
		return;
	}
	
	REAL origin[3];setVec3(origin, 0, 0, 0);
	REAL dBk_dPhik[16], dBk1_dPsik[16];	
	REAL Ak1_inv[16], Ak_inv[16];
	REAL tmp1[16], tmp2[16], tmp3[16];

	if(angle_k_idx>0){
		getRotationMatrixCalphaDPhi(dBk_dPhik, d_phi[angle_k_idx], d_psi[angle_k_idx-1], false);
	}else{
		getRotationMatrixCalphaDPhi(dBk_dPhik, d_phi[angle_k_idx], 0.0, true);
	}
	getRotationMatrixCalphaDPsi(dBk1_dPsik, d_phi[angle_k_idx+1], d_psi[angle_k_idx]);
		
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

__global__ void backwardFromCoordinates(REAL *angles, REAL *dr, REAL *dR_dangle, int *length, int angles_stride){
	int batch_idx = blockIdx.x;
	int batch_size = blockDim.x;
	int angle_idx = threadIdx.x;
	int num_angles = length[batch_idx];
	int num_atoms = num_angles;
	int atoms_stride = angles_stride;
	
	REAL *d_phi = angles + 2*batch_idx*angles_stride + angle_idx;
	REAL *d_psi = angles + (2*batch_idx+1)*angles_stride + angle_idx;
	
	REAL *dR_dPhi = dR_dangle + 2*batch_idx * (atoms_stride*angles_stride*3) + angle_idx*atoms_stride*3;
	REAL *dR_dPsi = dR_dangle + (2*batch_idx+1) * (atoms_stride*angles_stride*3) + angle_idx*atoms_stride*3;

	REAL *d_dr = dr + batch_idx*atoms_stride*3;

	for(int j=angle_idx; j<num_atoms; j++){
		
		(*d_phi) += vec3Mul(d_dr+3*j, dR_dPhi + 3*j);
		(*d_psi) += vec3Mul(d_dr+3*j, dR_dPsi + 3*j);
	}
}


void cpu_computeCoordinatesDihedral(REAL *angles, REAL *atoms, REAL *A, int *length, int batch_size, int angles_stride){
	computeCoordinatesDihedral<<<1,batch_size>>>(angles, atoms, A, length, angles_stride);
}

void cpu_computeDerivativesDihedral(REAL *angles, REAL *dR_dangle, REAL *A, int *length, int batch_size, int angles_stride){
	// dim3 angles_dim(angles_stride, angles_stride, 1);
	// computeGradientsOptimizedDihedral<<<batch_size, angles_dim>>>(angles, dR_dangle, A, length, angles_stride);
	
	//Time 1: 778ms
	// dim3 batch_angles_dim(batch_size, angles_stride, angles_stride);
	// computeGradientsOptimizedDihedral<<<batch_angles_dim, 1>>>(angles, dR_dangle, A, length, angles_stride);

	//Time 2: 43ms
	dim3 batch_angles_dim(batch_size, angles_stride, 1);
	computeGradientsOptimizedDihedral<<<batch_angles_dim, angles_stride>>>(angles, dR_dangle, A, length, angles_stride);
}

void cpu_backwardFromCoords(REAL *angles, REAL *dr, REAL *dR_dangle, int *length, int batch_size, int angles_stride){
	backwardFromCoordinates<<<batch_size, angles_stride>>>(angles, dr, dR_dangle, length, angles_stride);
}
