#include "cTensorProteinCUDAKernels.h"
#include "cMathCUDAKernels.cu"

__global__ void computeCoordinatesBackbone( REAL *angles, REAL *atoms, REAL *A, int *length, int angles_stride){
	uint batch_idx = threadIdx.x;
    int atoms_stride = 3*angles_stride;
    int num_angles = length[batch_idx];
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

__global__ void computeGradientsOptimizedBackbone( REAL *angles, REAL *dR_dangle, REAL *A, int *length, int angles_stride, bool norm){
	
	uint batch_size = blockDim.x;
	uint batch_idx = blockIdx.x;
	uint atom_i_idx = blockIdx.y;
    uint angle_k_idx = threadIdx.x;
			
	int num_angles = length[batch_idx];
	int num_atoms = 3*num_angles;
	int atoms_stride = 3*angles_stride;

	REAL *d_A = A + batch_idx * atoms_stride * 16;                

	REAL *d_phi = angles + 2*batch_idx*angles_stride;
	REAL *d_psi = angles + (2*batch_idx+1)*angles_stride;

	REAL *dR_dPhi = dR_dangle + 2*batch_idx * (atoms_stride*angles_stride*3) + angle_k_idx*atoms_stride*3 + atom_i_idx*3;
	REAL *dR_dPsi = dR_dangle + (2*batch_idx+1) * (atoms_stride*angles_stride*3) + angle_k_idx*atoms_stride*3 + atom_i_idx*3;
	
	REAL origin[3];setVec3(origin, 0, 0, 0);
	REAL dB3k1_dPhik[16], dB3k2_dPsik[16];
	REAL A3k1_inv[16], A3k2_inv[16];
	REAL tmp1[16], tmp2[16], tmp3[16];
    getRotationMatrixDihedralDPsi(dB3k1_dPhik, d_phi[angle_k_idx], C_N_CA, R_N_CA);
    getRotationMatrixDihedralDPsi(dB3k2_dPsik, d_psi[angle_k_idx], N_CA_C, R_CA_C);


	//dA_i / dphi_k
    if( (3*angle_k_idx+1) > atom_i_idx){
        setVec3(dR_dPhi, 0, 0, 0);
    }else{
        invertMat44(A3k1_inv, d_A + (3*angle_k_idx+1)*16);
        
        mat44Mul(d_A + (3*angle_k_idx)*16, dB3k1_dPhik, tmp1);
        mat44Mul(A3k1_inv, d_A + atom_i_idx*16, tmp2);
        mat44Mul(tmp1, tmp2, tmp3);
        mat44Vec3Mul(tmp3, origin, dR_dPhi);
		if(norm){
			REAL norm = getVec3Norm(dR_dPhi);
			if(norm>1E-5)vec3Mul(dR_dPhi, 1.0/norm);
		}

    }
	
	//dA_i / dpsi_k
    if( (3*angle_k_idx+2) > atom_i_idx){
        setVec3(dR_dPsi, 0, 0, 0);
    }else{
        invertMat44(A3k2_inv, d_A + (3*angle_k_idx + 2)*16);

        mat44Mul(d_A + (3*angle_k_idx+1)*16, dB3k2_dPsik, tmp1);
        mat44Mul(A3k2_inv, d_A + atom_i_idx*16, tmp2);
        mat44Mul(tmp1, tmp2, tmp3);
        mat44Vec3Mul(tmp3, origin, dR_dPsi);
		if(norm){
			REAL norm = getVec3Norm(dR_dPsi);
			if(norm>1E-5)vec3Mul(dR_dPsi, 1.0/norm);
		}
    }
	
}

__global__ void backwardFromCoordinatesBackbone(REAL *angles, REAL *dr, REAL *dR_dangle, int *length, int angles_stride){
	int batch_idx = blockIdx.x;
	int batch_size = blockDim.x;
	int angle_idx = threadIdx.x;
	int num_angles = length[batch_idx];
	int num_atoms = 3*num_angles;
	int atoms_stride = 3*angles_stride;
	
	REAL *d_phi = angles + 2*batch_idx*angles_stride + angle_idx;
	REAL *d_psi = angles + (2*batch_idx+1)*angles_stride + angle_idx;
	
	REAL *dR_dPhi = dR_dangle + 2*batch_idx * (atoms_stride*angles_stride*3) + angle_idx*atoms_stride*3;
	REAL *dR_dPsi = dR_dangle + (2*batch_idx+1) * (atoms_stride*angles_stride*3) + angle_idx*atoms_stride*3;

	REAL *d_dr = dr + batch_idx*atoms_stride*3;

	for(int j=3*angle_idx+2; j<num_atoms; j++){
		
		(*d_phi) += vec3Mul(d_dr+3*j, dR_dPhi + 3*j);
		(*d_psi) += vec3Mul(d_dr+3*j, dR_dPsi + 3*j);
	}
    
}


void cpu_computeCoordinatesBackbone(REAL *angles, REAL *atoms, REAL *A, int *length, int batch_size, int angles_stride){
	computeCoordinatesBackbone<<<1, batch_size>>>(angles, atoms, A, length, angles_stride);
}

void cpu_computeDerivativesBackbone(REAL *angles, REAL *dR_dangle, REAL *A, int *length, int batch_size, int angles_stride, bool norm){
	
	dim3 batch_angles_dim(batch_size, 3*angles_stride, 1);
	computeGradientsOptimizedBackbone<<<batch_angles_dim, angles_stride>>>(angles, dR_dangle, A, length, angles_stride, norm);
}

void cpu_backwardFromCoordsBackbone(REAL *angles, REAL *dr, REAL *dR_dangle, int *length, int batch_size, int angles_stride){
	backwardFromCoordinatesBackbone<<<batch_size, angles_stride>>>(angles, dr, dR_dangle, length, angles_stride);
}
