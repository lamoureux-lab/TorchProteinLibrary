#include "cMathCUDAKernels.cu"
#include <stdio.h>

#define WARP_SIZE 32

__global__ void computeCoordinatesBackbone( REAL *angles, REAL *atoms, REAL *A, int *length, int angles_stride){
	uint batch_idx = threadIdx.x;
    int atoms_stride = 3*angles_stride;
    int num_angles = length[batch_idx];
    int num_atoms = 3*length[batch_idx];
	// printf("angles_stride=%d, batch_idx=%d, num_atoms=%d, num_angles=%d\n",angles_stride, batch_idx,num_atoms,num_angles);

	REAL *d_atoms = atoms + batch_idx*(atoms_stride)*3;
	REAL *d_phi = angles + 3*batch_idx*angles_stride;
	REAL *d_psi = angles + (3*batch_idx+1)*angles_stride;
	REAL *d_omega = angles + (3*batch_idx+2)*angles_stride;
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
            getRotationMatrixDihedral(B, d_omega[angle_idx-1], CA_C_N, R_C_N);
        }
        
        mat44Mul(d_A+16*(i-1), B, d_A+16*(i));
	    mat44Vec3Mul(d_A+16*i, d_atoms, d_atoms + 3*i);
		// printf("batch_idx=%d, i=%d, x=%f, y=%f, z=%f\n", batch_idx, i, *(d_atoms + 3*i), *(d_atoms + 3*i+1), *(d_atoms + 3*i+2));
	}
}

__global__ void computeGradientsOptimizedBackboneOmega(REAL *angles, REAL *dR_dangle, REAL *A, int *length, int angles_stride){
	
	uint batch_size = blockDim.x;
	uint batch_idx = blockIdx.x;
	uint atom_i_idx = blockIdx.y;
	uint local_angle_k_idx = threadIdx.x;
    uint angle_k_idx = blockIdx.z*WARP_SIZE + local_angle_k_idx;
			
	int num_angles = length[batch_idx];
	int num_atoms = 3*num_angles;
	int atoms_stride = 3*angles_stride;
	
	REAL *d_A = A + batch_idx * atoms_stride * 16;
	REAL *d_A_warp = d_A + 3*blockIdx.z*WARP_SIZE*16;
	__shared__ REAL s_A [WARP_SIZE*3*16];
	for(int i=local_angle_k_idx; i<WARP_SIZE*3*16; i+=WARP_SIZE){
		if( (3*blockIdx.z*WARP_SIZE*16 + i)< (num_atoms*16) )
			s_A[i] = d_A_warp[i];	
	}
	__syncthreads();

	if(angle_k_idx>=angles_stride)return;
	REAL *d_omega = angles + (3*batch_idx+2)*angles_stride;
	REAL *dR_dOmega = dR_dangle + (3*batch_idx+2) * (atoms_stride*angles_stride*3) + atom_i_idx*angles_stride*3 + angle_k_idx*3;
	REAL tmp1[16], tmp2[16], tmp3[16];
    
	//dA_i / dpsi_k
    if( (3*angle_k_idx) > atom_i_idx || (3*angle_k_idx == 0)){
        setVec3(dR_dOmega, 0, 0, 0);
    }else{
        getRotationMatrixDihedralDPsi(tmp2, d_omega[angle_k_idx], CA_C_N, R_C_N);
        mat44Mul(s_A + (3*local_angle_k_idx-1)*16, tmp2, tmp1);
        
		invertMat44(tmp3, s_A + (3*local_angle_k_idx)*16);
		mat44Mul(tmp3, d_A + atom_i_idx*16, tmp2);
		
        mat44Mul(tmp1, tmp2, tmp3);
        mat44Zero3Mul(tmp3, dR_dOmega);
	}
	
}

__device__ void device_singleAngleAtom(	REAL *d_angle, //pointer to the angle stride
										REAL *dR_dAngle, //pointer to the gradient
										REAL *d_A, //pointer to the atom transformation matrix batch
										REAL *d_A_warp, //shared pointer to the atom transformation matrix
										int angle_k, //angle index (phi:0, psi:1, omega:2)
										int angle_idx, //angle index
										int angle_widx, //angle index in the warp
										int atom_idx //atom index
){
	REAL tmp1[16], tmp2[16], tmp3[16];
	REAL B[16], Ar_inv[16];
	REAL bond_angles[] = {C_N_CA, N_CA_C, CA_C_N};
    REAL bond_lengths[] = {R_N_CA, R_CA_C, R_C_N};

	if( (3*angle_idx+angle_k) > atom_idx){
        setVec3(dR_dAngle, 0, 0, 0);
    }else{
        getRotationMatrixDihedralDPsi(B, d_angle[angle_idx], bond_angles[angle_k], bond_lengths[angle_k]);
        mat44Mul(d_A_warp + (3*angle_widx + angle_k)*16, B, tmp1);
        
		invertMat44(Ar_inv, d_A_warp + (3*angle_widx + angle_k + 1)*16);
		mat44Mul(Ar_inv, d_A + atom_idx*16, tmp2);
		
        mat44Mul(tmp1, tmp2, tmp3);
        mat44Zero3Mul(tmp3, dR_dAngle);
	}
}

__global__ void computeGradientsOptimizedBackbone( REAL *angles, REAL *dR_dangle, REAL *A, int *length, int angles_stride){
	uint batch_idx = blockIdx.x;
	uint atom_idx = blockIdx.y;
	uint angle_widx = threadIdx.x;
	uint warp_idx = blockIdx.z;
    uint angle_idx = warp_idx*WARP_SIZE + angle_widx;

	int num_angles = length[batch_idx];
	int num_atoms = 3*num_angles;
	int atoms_stride = 3*angles_stride;

	REAL *d_A = A + batch_idx * atoms_stride * 16;
	REAL *d_A_warp = d_A + 3*warp_idx*WARP_SIZE * 16;
	__shared__ REAL s_A[(3*WARP_SIZE + 1)*16];
		
	for(int i=threadIdx.x; i<(3*WARP_SIZE + 1)*16; i+=WARP_SIZE){
		if( ((3*warp_idx*WARP_SIZE + 1)*16 + i) < (num_atoms*16)){
			s_A[i] = d_A_warp[i];
		}
	}
	__syncthreads();
	
	if(angle_idx>=angles_stride)return;

	for(int angle_k=0; angle_k<3; angle_k++){
		device_singleAngleAtom(	angles + (3*batch_idx + angle_k) * angles_stride,
								dR_dangle + (3*batch_idx + angle_k) * (atoms_stride*angles_stride*3) + angle_idx*atoms_stride*3 + atom_idx*3,
								d_A,
								s_A,
								angle_k, angle_idx, angle_widx, atom_idx);
	}
	
}

__global__ void backwardFromCoordinatesBackbone(REAL *angles, REAL *dr, REAL *dR_dangle, int *length, int angles_stride){
	int batch_idx = blockIdx.x;
	int batch_size = blockDim.x;
	int angle_k = blockIdx.y;
	int angle_idx = threadIdx.x;
	int num_angles = length[batch_idx];
	int num_atoms = 3*num_angles;
	int atoms_stride = 3*angles_stride;
	
	REAL *d_angle = angles + (3*batch_idx + angle_k)*angles_stride + angle_idx;
	REAL *dR_dAngle = dR_dangle + (3*batch_idx + angle_k) * (atoms_stride*angles_stride*3) + angle_idx*atoms_stride*3;
	REAL *d_dr = dr + batch_idx*atoms_stride*3;

	for(int j=3*angle_idx; j<num_atoms; j++){
		(*d_angle) += vec3Mul(d_dr + 3*j, dR_dAngle + j*3);
	}
}

void gpu_computeCoordinatesBackbone(REAL *angles, REAL *atoms, REAL *A, int *length, int batch_size, int angles_stride){
	computeCoordinatesBackbone<<<1, batch_size>>>(angles, atoms, A, length, angles_stride);
}

void gpu_computeDerivativesBackbone(REAL *angles, REAL *dR_dangle, REAL *A, int *length, int batch_size, int angles_stride){
	dim3 batch_angles_dim_special(batch_size, 3*angles_stride, angles_stride/WARP_SIZE + 1);
	computeGradientsOptimizedBackbone<<<batch_angles_dim_special, WARP_SIZE>>>(angles, dR_dangle, A, length, angles_stride);
}

void gpu_backwardFromCoordsBackbone(REAL *angles, REAL *dr, REAL *dR_dangle, int *length, int batch_size, int angles_stride){
	dim3 batch_angles_dim_special(batch_size, 3, 1);
	backwardFromCoordinatesBackbone<<<batch_angles_dim_special, angles_stride>>>(angles, dr, dR_dangle, length, angles_stride);
}

