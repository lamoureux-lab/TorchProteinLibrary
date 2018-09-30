#include "cRMSD.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <tuple>
#include <limits>

cRMSD::cRMSD(uint num_atoms){
    this->num_atoms = num_atoms;
    centroid_src.setZero();
    centroid_dst.setZero();

    ce_src = new double [3*num_atoms];
    ce_dst = new double [3*num_atoms];
    U_ce_src = new double [3*num_atoms];
    UT_ce_dst = new double [3*num_atoms];
    external = false;
}

cRMSD::cRMSD(double *ce_src, double *ce_dst, double *U_ce_src, double *UT_ce_dst, const uint num_atoms){
    this->num_atoms = num_atoms;
    centroid_src.setZero();
    centroid_dst.setZero();

    this->ce_src = ce_src;
    this->ce_dst = ce_dst;
    this->U_ce_src = U_ce_src;
    this->UT_ce_dst = UT_ce_dst;
    external = true;
}

cRMSD::~cRMSD(){
    if(!external){
        delete [] ce_src;
        delete [] ce_dst;
        delete [] U_ce_src;
        delete [] UT_ce_dst;
    }
}

double cRMSD::compute( double *src, double *dst ){        
    //computing centroids
    for(int i=0; i<num_atoms; i++){
        centroid_src += cVector3(src+3*i);
        centroid_dst += cVector3(dst+3*i);
    }
    centroid_src/=num_atoms; centroid_dst/=num_atoms;
    
    //centering the proteins
    for(int i=0; i<num_atoms; i++){
        cVector3 ce_src_atom(ce_src+3*i);
        cVector3 ce_dst_atom(ce_dst+3*i);
        ce_src_atom = cVector3(src+3*i) - centroid_src;
        ce_dst_atom = cVector3(dst+3*i) - centroid_dst;
    }

    //correlation matrix R
    cMatrix33 R;
    for(int i=0; i<num_atoms; i++){
        cVector3 ce_src_atom(ce_src+3*i);
        cVector3 ce_dst_atom(ce_dst+3*i);
        R.m[0][0] += ce_src_atom.v[0]*ce_dst_atom.v[0];
        R.m[0][1] += ce_src_atom.v[0]*ce_dst_atom.v[1];
        R.m[0][2] += ce_src_atom.v[0]*ce_dst_atom.v[2];
        R.m[1][0] += ce_src_atom.v[1]*ce_dst_atom.v[0];
        R.m[1][1] += ce_src_atom.v[1]*ce_dst_atom.v[1];
        R.m[1][2] += ce_src_atom.v[1]*ce_dst_atom.v[2];
        R.m[2][0] += ce_src_atom.v[2]*ce_dst_atom.v[0];
        R.m[2][1] += ce_src_atom.v[2]*ce_dst_atom.v[1];
        R.m[2][2] += ce_src_atom.v[2]*ce_dst_atom.v[2];
    }

    //Converting R to the T matrix
    at::Tensor Tmat = at::CPU(at::kDouble).zeros({4,4});
    double *T = Tmat.data<double>();
    T[0] = R.m[0][0]+R.m[1][1]+R.m[2][2];T[1] = R.m[1][2]-R.m[2][1];T[2] = R.m[2][0]-R.m[0][2];T[3] = R.m[0][1]-R.m[1][0];
    T[4] = R.m[1][2]-R.m[2][1];T[5] = R.m[0][0]-R.m[1][1]-R.m[2][2];T[6] = R.m[0][1]+R.m[1][0];T[7] = R.m[0][2]+R.m[2][0];
    T[8] = R.m[2][0]-R.m[0][2];T[9] = R.m[0][1]+R.m[1][0];T[10] = -R.m[0][0]+R.m[1][1]-R.m[2][2];T[11] = R.m[1][2]+R.m[2][1];
    T[12] = R.m[0][1]-R.m[1][0];T[13] = R.m[0][2]+R.m[2][0];T[14] = R.m[1][2]+R.m[2][1];T[15] = -R.m[0][0]-R.m[1][1]+R.m[2][2];
        
    //soving eigenvalues problem
    // THDoubleTensor_syev(eig, vec, Tmat, "V", "U");
    std::tuple<at::Tensor, at::Tensor> result = Tmat.eig(true);
    
    //getting maximum eigenvalue and eigenvector
    double max_eig_val = std::numeric_limits<double>::min();
    at::Tensor max_eig_vec = at::CPU(at::kDouble).zeros({4});
    auto q = max_eig_vec.accessor<double, 1>();

    at::Tensor eig_vals = std::get<0>(result);
    at::Tensor eig_vecs = std::get<1>(result);
    auto eig_val = eig_vals.accessor<double, 2>();
    for(int i=0; i<4; i++){    
        if(max_eig_val < eig_val[i][0]){
            max_eig_val = eig_val[i][0];        
            for(int j=0;j<4; j++)max_eig_vec[j] = eig_vecs[j][i];
        }
    }
    
        
    // rotation matrix
    U.m[0][0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
    U.m[0][1] = 2.0*(q[1]*q[2] - q[0]*q[3]);
    U.m[0][2] = 2.0*(q[1]*q[3] + q[0]*q[2]);

    U.m[1][0] = 2.0*(q[1]*q[2] + q[0]*q[3]);
    U.m[1][1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
    U.m[1][2] = 2.0*(q[2]*q[3] - q[0]*q[1]);

    U.m[2][0] = 2.0*(q[1]*q[3] - q[0]*q[2]);
    U.m[2][1] = 2.0*(q[2]*q[3] + q[0]*q[1]);
    U.m[2][2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];

    UT = U.getTranspose();
    
    //computing R2 coefficient
    double R2 = 0.0;
    for(int i=0; i<num_atoms; i++){
        cVector3 ce_src_atom(ce_src+3*i);
        cVector3 ce_dst_atom(ce_dst+3*i);
        R2 += ce_src_atom.norm2() + ce_dst_atom.norm2();
    }
    
    double rmsd = (R2 - 2.0*fabs(max_eig_val))/(double(num_atoms));
    
    //transforming coordinates
    for(int i=0; i<num_atoms; i++){
        cVector3 ce_src_atom(ce_src+3*i);
        cVector3 ce_dst_atom(ce_dst+3*i);
        cVector3 U_src_atom(U_ce_src+3*i);
        cVector3 UT_dst_atom(UT_ce_dst+3*i);
        U_src_atom = U*ce_src_atom;
        UT_dst_atom = UT*ce_dst_atom;
    } 
    
    return rmsd;
}

void cRMSD::grad( double *grad_atoms, double *grad_output ){
    float mult = grad_output[0]/(double(num_atoms));
    for(int i=0;i<num_atoms;i++){
        cVector3 grad_src_atom(grad_atoms+3*i);
        cVector3 src_atom(ce_src+3*i);
        cVector3 UT_dst_atom(UT_ce_dst+3*i);
        grad_src_atom = (src_atom - UT_dst_atom)*mult;
    }
}