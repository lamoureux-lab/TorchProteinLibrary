#include "cBackboneProteinCUDAKernels.h"
#include "cMathCUDAKernels.cu"

#define WARP_SIZE 32

__global__ void computeCoordinatesBackbone( REAL *angles, REAL *atoms, REAL *A, int *length, int angles_stride){
  uint batch_idx = threadIdx.x;
  int atoms_stride = 3*angles_stride;
  int num_atoms = 3*length[batch_idx];

  REAL *d_atoms = atoms + batch_idx*(atoms_stride)*3;
  REAL *d_phi = angles + 2*batch_idx*angles_stride;
  REAL *d_psi = angles + (2*batch_idx+1)*angles_stride;
  REAL *d_A = A + batch_idx*atoms_stride*16;

  REAL B[16];
  int angle_idx = 0;

  //N atom
  setVec3(d_atoms, 0.0, 0.0, 0.0);
  getIdentityMatrix44(d_A);

  for(int i=1; i<num_atoms; i++){
    angle_idx = i/3;
    if(i%3 == 1){
      getRotationMatrixDihedral(B, d_phi[angle_idx], C_N_CA, R_N_CA);
    }else if (i%3 == 2){
      getRotationMatrixDihedral(B, d_psi[angle_idx], N_CA_C, R_CA_C);
    }else{
      getRotationMatrixDihedral(B, OMEGACIS, CA_C_N, R_C_N);
    }

    mat44Mul(d_A+16*(i-1), B, d_A+16*(i));
    mat44Vec3Mul(d_A+16*i, d_atoms, d_atoms + 3*i);
  }
}

__global__ void computeGradientsOptimizedBackbone( REAL *angles, REAL *dR_dangle, REAL *A, int *length, int angles_stride){

  volatile uint batch_size = blockDim.x;
  volatile uint batch_idx = blockIdx.x;
  volatile uint atom_i_idx = blockIdx.y;
  volatile uint angle_k_idx = threadIdx.x;

  volatile int num_angles = length[batch_idx];
  volatile int num_atoms = 3*num_angles;
  volatile int atoms_stride = 3*angles_stride;

  REAL *d_A = A + batch_idx * atoms_stride * 16;

  REAL *d_phi = angles + 2*batch_idx*angles_stride;
  REAL *d_psi = angles + (2*batch_idx+1)*angles_stride;

  REAL *dR_dPhi = dR_dangle + 2*batch_idx * (atoms_stride*angles_stride*3) + atom_i_idx*angles_stride*3 + angle_k_idx*3;
  REAL *dR_dPsi = dR_dangle + (2*batch_idx+1) * (atoms_stride*angles_stride*3) + atom_i_idx*angles_stride*3 + angle_k_idx*3;

  REAL origin[3];setVec3(origin, 0, 0, 0);
  REAL dB_dangle[16], A_inv[16];
  REAL tmp1[16], tmp2[16], tmp3[16];

  //dA_i / dphi_k
  if( (3*angle_k_idx+1) > atom_i_idx){
    setVec3(dR_dPhi, 0, 0, 0);
  }else{
    invertMat44(A_inv, d_A + (3*angle_k_idx+1)*16);
    getRotationMatrixDihedralDPsi(dB_dangle, d_phi[angle_k_idx], C_N_CA, R_N_CA);
    mat44Mul(d_A + (3*angle_k_idx)*16, dB_dangle, tmp1);
    mat44Mul(A_inv, d_A + atom_i_idx*16, tmp2);
    mat44Mul(tmp1, tmp2, tmp3);
    mat44Vec3Mul(tmp3, origin, dR_dPhi);
  }

  //dA_i / dpsi_k
  if( (3*angle_k_idx+2) > atom_i_idx){
    setVec3(dR_dPsi, 0, 0, 0);
  }else{
    invertMat44(A_inv, d_A + (3*angle_k_idx + 2)*16);
    getRotationMatrixDihedralDPsi(dB_dangle, d_psi[angle_k_idx], N_CA_C, R_CA_C);
    mat44Mul(d_A + (3*angle_k_idx+1)*16, dB_dangle, tmp1);
    mat44Mul(A_inv, d_A + atom_i_idx*16, tmp2);
    mat44Mul(tmp1, tmp2, tmp3);
    mat44Vec3Mul(tmp3, origin, dR_dPsi);
  }
}

__global__ void computeGradientsOptimizedBackbonePhi( REAL *angles, REAL *dR_dangle, REAL *A, int *length, int angles_stride){

  uint batch_idx = blockIdx.x;
  uint atom_i_idx = blockIdx.y;
  uint local_angle_k_idx = threadIdx.x;
  uint angle_k_idx = blockIdx.z*WARP_SIZE + local_angle_k_idx;

  int atoms_stride = 3*angles_stride;

  REAL *d_A = A + batch_idx * atoms_stride * 16;
  REAL *d_A_warp = d_A + 3*blockIdx.z*WARP_SIZE*16;
  __shared__ REAL s_A[WARP_SIZE*3*16];

  for(int i=local_angle_k_idx; i<WARP_SIZE*3*16; i+=WARP_SIZE){
    s_A[i] = d_A_warp[i];
  }
  __syncthreads();

  if(angle_k_idx>=angles_stride)return;
  REAL *d_phi = angles + 2*batch_idx*angles_stride;
  REAL *dR_dPhi = dR_dangle + 2*batch_idx * (atoms_stride*angles_stride*3) + atom_i_idx*angles_stride*3 + angle_k_idx*3;
  REAL tmp1[16], tmp2[16], tmp3[16];

  //dA_i / dphi_k
  if( (3*angle_k_idx+1) > atom_i_idx){
    setVec3(dR_dPhi, 0, 0, 0);
  }else{
    getRotationMatrixDihedralDPsi(tmp2, d_phi[angle_k_idx], C_N_CA, R_N_CA);
    mat44Mul(&s_A[3*local_angle_k_idx*16], tmp2, tmp1);

    invertMat44(tmp3, &s_A[(3*local_angle_k_idx+1)*16]);
    mat44Mul(tmp3, d_A + atom_i_idx*16, tmp2);

    mat44Mul(tmp1, tmp2, tmp3);
    mat44Zero3Mul(tmp3, dR_dPhi);

  }
}

__global__ void computeGradientsOptimizedBackbonePsi( REAL *angles, REAL *dR_dangle, REAL *A, int *length, int angles_stride){

  uint batch_idx = blockIdx.x;
  uint atom_i_idx = blockIdx.y;
  uint local_angle_k_idx = threadIdx.x;
  uint angle_k_idx = blockIdx.z*WARP_SIZE + local_angle_k_idx;

  int atoms_stride = 3*angles_stride;

  REAL *d_A = A + batch_idx * atoms_stride * 16;
  REAL *d_A_warp = d_A + 3*blockIdx.z*WARP_SIZE*16;
  __shared__ REAL s_A [WARP_SIZE*3*16];
  for(int i=local_angle_k_idx; i<WARP_SIZE*3*16; i+=WARP_SIZE){
    s_A[i] = d_A_warp[i];
  }
  __syncthreads();

  if(angle_k_idx>=angles_stride)return;
  REAL *d_psi = angles + (2*batch_idx+1)*angles_stride;
  REAL *dR_dPsi = dR_dangle + (2*batch_idx+1) * (atoms_stride*angles_stride*3) + atom_i_idx*angles_stride*3 + angle_k_idx*3;
  REAL tmp1[16], tmp2[16], tmp3[16];

  //dA_i / dpsi_k
  if( (3*angle_k_idx+2) > atom_i_idx){
    setVec3(dR_dPsi, 0, 0, 0);
  }else{
    getRotationMatrixDihedralDPsi(tmp2, d_psi[angle_k_idx], N_CA_C, R_CA_C);
    mat44Mul(s_A + (3*local_angle_k_idx+1)*16, tmp2, tmp1);

    invertMat44(tmp3, s_A + (3*local_angle_k_idx + 2)*16);
    mat44Mul(tmp3, d_A + atom_i_idx*16, tmp2);

    mat44Mul(tmp1, tmp2, tmp3);
    mat44Zero3Mul(tmp3, dR_dPsi);
  }
}


__global__ void backwardFromCoordinatesBackbone(REAL *angles, const REAL *dr, const REAL *dR_dangle, const int *length, int angles_stride, bool norm){
  int batch_idx = blockIdx.x;
  int angle_idx = threadIdx.x;
  int num_angles = length[batch_idx];
  int num_atoms = 3*num_angles;
  int atoms_stride = 3*angles_stride;

  REAL *d_phi = angles + 2*batch_idx*angles_stride + angle_idx;
  REAL *d_psi = angles + (2*batch_idx+1)*angles_stride + angle_idx;

  const REAL *dR_dPhi = dR_dangle + 2*batch_idx * (atoms_stride*angles_stride*3);
  const REAL *dR_dPsi = dR_dangle + (2*batch_idx+1) * (atoms_stride*angles_stride*3);

  const REAL *d_dr = dr + batch_idx*atoms_stride*3;

  for (int j=3*angle_idx+2; j<num_atoms; j++) {
    (*d_phi) += vec3Mul(d_dr+3*j, dR_dPhi + j*angles_stride*3 + angle_idx*3);
    (*d_psi) += vec3Mul(d_dr+3*j, dR_dPsi + j*angles_stride*3 + angle_idx*3);
  }
  if (norm) {
    if (abs((*d_phi))>10.0) {
      (*d_phi) = copysignf(1, (*d_phi))*10.0;
    }
    if (abs((*d_psi))>10.0){
      (*d_psi) = copysignf(1, (*d_psi))*10.0;
    }
  }
}
/*
__global__ void backwardFromCoordinatesBackbonePhi(REAL *angles, REAL *dr, REAL *dR_dangle, int *length, int angles_stride, bool norm){
	int batch_idx = blockIdx.x;
	int batch_size = blockDim.x;
	int local_angle_idx = threadIdx.x;
	int global_angle_idx = blockIdx.y*WARP_SIZE + threadIdx.x;
	int num_angles = length[batch_idx];
	int num_atoms = 3*num_angles;
	int atoms_stride = 3*angles_stride;

	REAL *d_phi = angles + 2*batch_idx*angles_stride + global_angle_idx;
	REAL *dR_dPhi = dR_dangle + 2*batch_idx * (atoms_stride*angles_stride*3);
	REAL *d_dr = dr + batch_idx*atoms_stride*3;
	REAL mag;

	for(int j=3*angle_idx+2; j<num_atoms; j++){
		(*d_phi) += vec3Mul(d_dr+3*j, dR_dPhi + j*angles_stride*3 + angle_idx*3);
	}

	if(norm){
		if (abs((*d_phi))>10.0){
			(*d_phi) = copysignf(1, (*d_phi))*10.0;
		}
		if (abs((*d_psi))>10.0){
			(*d_psi) = copysignf(1, (*d_psi))*10.0;
		}
	}

}
*/

void cpu_computeCoordinatesBackbone(REAL *angles, REAL *atoms, REAL *A, int *length, int batch_size, int angles_stride){
	computeCoordinatesBackbone<<<1, batch_size>>>(angles, atoms, A, length, angles_stride);
}

void cpu_computeDerivativesBackbone(REAL *angles, REAL *dR_dangle, REAL *A, int *length, int batch_size, int angles_stride){
	dim3 batch_angles_dim_special(batch_size, 3*angles_stride, angles_stride/WARP_SIZE + 1);
	computeGradientsOptimizedBackbonePhi<<<batch_angles_dim_special, WARP_SIZE>>>(angles, dR_dangle, A, length, angles_stride);
	computeGradientsOptimizedBackbonePsi<<<batch_angles_dim_special, WARP_SIZE>>>(angles, dR_dangle, A, length, angles_stride);
}

void cpu_backwardFromCoordsBackbone(REAL *angles, REAL *dr, REAL *dR_dangle, int *length, int batch_size, int angles_stride, bool norm){
	backwardFromCoordinatesBackbone<<<batch_size, angles_stride>>>(angles, dr, dR_dangle, length, angles_stride, norm);
	// dim3 batch_angles_dim_special(batch_size, 3*angles_stride, angles_stride/WARP_SIZE + 1);
	// backwardFromCoordinatesBackbonePhi<<<batch_size, angles_stride>>>(angles, dr, dR_dangle, length, angles_stride, norm);
}
