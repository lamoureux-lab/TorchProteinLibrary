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
            angle_idx = int(i/3);
            cMatrix44 B;
            if(i%3 == 1){
                B.setDihedral(d_phi[angle_idx], C_N_CA, R_N_CA);
            }else if (i%3 == 2){
                B.setDihedral(d_psi[angle_idx], N_CA_C, R_CA_C);
            }else{
                B.setDihedral(d_omega[angle_idx-1], CA_C_N, R_C_N);
            }
            cMatrix44 Aim1(d_A+16*(i-1));
            cMatrix44 Ai(d_A+16*(i));
            cVector3 ri(d_atoms + 3*i);

            Ai = Aim1 * B;
            ri = Ai * r0;
        }
    }
}

void device_singleAngleAtom(double *d_angle, //pointer to the angle stride
                            double *dR_dAngle, //pointer to the gradient
                            double *d_A, //pointer to the atom transformation matrix batch
                            int angle_k, //angle index (phi:0, psi:1, omega:2)
                            int angle_idx, //angle index
                            int atom_idx //atom index
){
/*
Computes derivative of atom "atom_idx" coordinates with respect to {phi, psi, omega}[angle_k] with the index "angle_idx".
*/
    
    double bond_angles[] = {C_N_CA, N_CA_C, CA_C_N};
    double bond_lengths[] = {R_N_CA, R_CA_C, R_C_N};
    cVector3 zero; zero.setZero();
    cVector3 grad(dR_dAngle);
    if( (3*angle_idx+angle_k) > atom_idx){
        grad.setZero();
    }else{
        
        cMatrix44 B, Ar_inv;
        cMatrix44 Al(d_A + (3*angle_idx + angle_k)*16), Ar(d_A + (3*angle_idx + angle_k + 1)*16);

        B.setDihedralDphi(d_angle[angle_idx], bond_angles[angle_k], bond_lengths[angle_k]);
        Ar_inv = invertTransform44(Ar);
        cMatrix44 Aj(d_A + atom_idx*16);
        
        grad = (Al * B * Ar_inv * Aj) * zero;
    }
}

void cpu_computeDerivativesBackbone(    double *angles,  
                                        double *dR_dangle,   
                                        double *A,       
                                        int *length,
                                        int batch_size,
                                        int angles_stride){
    int atoms_stride = 3*angles_stride;
    double *d_angle, *dR_dAngle;
    for(int batch_idx=0; batch_idx<batch_size; batch_idx++){
        int num_angles = length[batch_idx];
        int num_atoms = 3*num_angles;
        double *d_A = A + batch_idx * atoms_stride * 16;

        for(int atom_idx=0; atom_idx<num_atoms; atom_idx++){
            for(int angle_idx=0; angle_idx<num_angles; angle_idx++){               	            
                for(int angle_k=0; angle_k<3; angle_k++){                
                    device_singleAngleAtom( angles + (3*batch_idx+angle_k)*angles_stride, 
                                            dR_dangle + (3*batch_idx+angle_k) * (atoms_stride*angles_stride*3) + angle_idx*(atoms_stride*3) + atom_idx*3,
                                            d_A, angle_k, angle_idx, atom_idx);
                }
            }
        }
    }
};

void cpu_backwardFromCoordsBackbone(    double *gradInput,
                                        double *gradOutput,
                                        double *dR_dangle,
                                        int *length,
                                        int batch_size,
                                        int angles_stride){
    
    int atoms_stride = 3*angles_stride;
    for(int batch_idx=0; batch_idx<batch_size; batch_idx++){
        int num_angles = length[batch_idx];
        for(int angle_idx=0; angle_idx<num_angles; angle_idx++){
            int num_atoms = 3*num_angles;
            for(int angle_k=0; angle_k<3; angle_k++){
    
                double *d_angle = gradInput + (3*batch_idx+angle_k) * angles_stride + angle_idx;
                double *dR_dAngle = dR_dangle + (3*batch_idx+angle_k) * (atoms_stride*angles_stride*3) + angle_idx*atoms_stride*3;

                for(int atom_idx=3*angle_idx; atom_idx<num_atoms; atom_idx++){
                    cVector3 dr(gradOutput + batch_idx*atoms_stride*3 + 3*atom_idx);
                    cVector3 dr_dangle(dR_dAngle + atom_idx*3);
                    (*d_angle) += dr | dr_dangle;
                }
            }
        }
    }
};