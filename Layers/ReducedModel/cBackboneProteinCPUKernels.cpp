#include "cBackboneProteinCPUKernels.hpp"

#include <cVector3.h>
#include <cMatrix44.h>

#define R_CA_C 1.525
#define R_C_N 1.330
#define R_N_CA 1.460

#define CA_C_N (M_PI - 2.1186)
#define C_N_CA (M_PI - 1.9391)
#define N_CA_C (M_PI - 2.061)


void cpu_computeCoordinatesBackbone(    double *angles, 
                                    double *atoms, 
                                    double *A, 
                                    int *length, 
                                    int batch_size, 
                                    int angles_stride){
    int atoms_stride = 3*angles_stride;
    
    for(int batch_idx=0; batch_idx<batch_size; batch_idx++){
        int num_angles = length[batch_idx];
        int num_atoms = 3*num_angles;

        double *d_atoms = atoms + batch_idx*(atoms_stride)*3;
        double *d_phi = angles + 3*batch_idx*angles_stride;
        double *d_psi = angles + (3*batch_idx+1)*angles_stride;
        double *d_omega = angles + (3*batch_idx+2)*angles_stride;
        double *d_A = A + batch_idx*atoms_stride*16;
            
        double B[16];
        int angle_idx = 0;

        //N atom
        cVector3 r0(d_atoms);
        r0.setZero();
        cMatrix44 A(d_A);
        A.setIdentity();
                
        for(int i=1; i<num_atoms; i++){
            angle_idx = i/3;
            cMatrix44 B;
            if(i%3 == 1){
                B.setDihedral(d_phi[angle_idx], C_N_CA, R_N_CA);
            }else if (i%3 == 2){
                B.setDihedral(d_psi[angle_idx], N_CA_C, R_CA_C);
            }else{
                B.setDihedral(d_omega[angle_idx], CA_C_N, R_C_N);
            }
            cMatrix44 Aim1(d_A+16*(i-1));
            cMatrix44 Ai(d_A+16*(i));
            cVector3 ri(d_atoms + 3*i);

            Ai = Aim1 * B;
            ri = Ai * r0;
        }
    }

}

void cpu_computeDerivativesBackbone(    double *angles,  
									double *dR_dangle,   
									double *A,       
									int *length,
									int batch_size,
									int angles_stride){
    int atoms_stride = 3*angles_stride;
    cVector3 zero; zero.setZero();
    for(int batch_idx=0; batch_idx<batch_size; batch_idx++){
        int num_angles = length[batch_idx];
        int num_atoms = 3*num_angles;
        for(int atom_idx=0; atom_idx<num_atoms; atom_idx++){
            for(int angle_idx=0; angle_idx<num_angles; angle_idx++){
                	            
                double *d_A = A + batch_idx * atoms_stride * 16 + angle_idx*16;
                double *d_angle, *dR_dAngle;

                
                //PHI
                d_angle = angles + (3*batch_idx)*angles_stride;
                dR_dAngle = dR_dangle + (3*batch_idx) * (atoms_stride*angles_stride*3) + atom_idx * (angles_stride*3) + angle_idx*3;
                cVector3 dR_dPhi(dR_dAngle);
                if( (3*angle_idx+1) > atom_idx){
                    dR_dPhi.setZero();
                }else{
                    cMatrix44 dB3jp1, A3jp1_inv;
                    cMatrix44 A3j(d_A + 3*angle_idx*16), A3jp1(d_A + (3*angle_idx+1)*16);

                    dB3jp1.setDihedralDphi(d_angle[angle_idx], C_N_CA, R_N_CA);
                    A3jp1_inv = invertTransform44(A3jp1);
                    cMatrix44 Ai(d_A + atom_idx*16);
                    
                    dR_dPhi = (A3j * dB3jp1 * A3jp1_inv * Ai) * zero;
                }
                
                //PSI
                d_angle = angles + (3*batch_idx+1)*angles_stride;
                dR_dAngle = dR_dangle + (3*batch_idx+1) * (atoms_stride*angles_stride*3) + atom_idx*angles_stride*3 + angle_idx*3;
                cVector3 dR_dPsi(dR_dAngle);
                if( (3*angle_idx+2) > atom_idx){
                    dR_dPsi.setZero();
                }else{
                    cMatrix44 dB3jp2, A3jp2_inv;
                    cMatrix44 A3jp1(d_A + (3*angle_idx+1)*16), A3jp2(d_A + (3*angle_idx+2)*16);

                    dB3jp2.setDihedralDphi(d_angle[angle_idx], N_CA_C, R_CA_C);
                    A3jp2_inv = invertTransform44(A3jp2);
                    cMatrix44 Ai(d_A + atom_idx*16);
                    
                    dR_dPsi = (A3jp1 * dB3jp2 * A3jp2_inv * Ai) * zero;
                }

                //OMEGA
                d_angle = angles + (3*batch_idx+2)*angles_stride;
                dR_dAngle = dR_dangle + (3*batch_idx+2) * (atoms_stride*angles_stride*3) + atom_idx*angles_stride*3 + angle_idx*3;
                cVector3 dR_dOmega(dR_dAngle);
                if( (3*angle_idx) > atom_idx || (angle_idx == 0)){
                    dR_dOmega.setZero();
                }else{
                    cMatrix44 dB3j, A3j_inv;
                    cMatrix44 A3jm1(d_A + (3*angle_idx-1)*16), A3j(d_A + (3*angle_idx)*16);

                    dB3j.setDihedralDphi(d_angle[angle_idx], CA_C_N, R_C_N);
                    A3j_inv = invertTransform44(A3j);
                    cMatrix44 Ai(d_A + atom_idx*16);
                    
                    dR_dOmega = (A3jm1 * dB3j * A3j_inv * Ai) * zero;
                }
            }
        }
    }
};

void cpu_backwardFromCoordsBackbone(    double *dangles,
									double *dr,
									double *dR_dangle,
									int *length,
									int batch_size,
									int angles_stride){
    
    int atoms_stride = 3*angles_stride;
    for(int batch_idx=0; batch_idx<batch_size; batch_idx++){
        int num_angles = length[batch_idx];
        for(int angle_idx=0; angle_idx<num_angles; angle_idx++){
            int num_atoms = 3*num_angles;
    
            double *d_phi = dangles + 3*batch_idx*angles_stride + angle_idx;
            double *d_psi = dangles + (3*batch_idx+1)*angles_stride + angle_idx;
            double *d_omega = dangles + (3*batch_idx+2)*angles_stride + angle_idx;
            
            double *dR_dPhi = dR_dangle + 3*batch_idx * (atoms_stride*angles_stride*3);
            double *dR_dPsi = dR_dangle + (3*batch_idx+1) * (atoms_stride*angles_stride*3);
            double *dR_dOmega = dR_dangle + (3*batch_idx+2) * (atoms_stride*angles_stride*3);

            for(int j=3*angle_idx; j<num_atoms; j++){
                cVector3 dr(dr + batch_idx*atoms_stride*3 + 3*j);
                cVector3 dr_dphi(dR_dPhi + j*angles_stride*3 + angle_idx*3);
                cVector3 dr_dpsi(dR_dPsi + j*angles_stride*3 + angle_idx*3);
                cVector3 dr_domega(dR_dOmega + j*angles_stride*3 + angle_idx*3);
                (*d_phi) += dr | dr_dphi;
                (*d_psi) += dr | dr_dpsi;
                (*d_omega) += dr | dr_domega;
            }
        }
    }

};