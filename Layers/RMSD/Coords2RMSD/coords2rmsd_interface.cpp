#include "coords2rmsd_interface.h"
#include "../RMSDKernels.h"
#include <iostream>
#include <tuple>
#include <limits>
#include <nUtil.h>
#include <math.h>


#define EPS_LIMIT 1E-10

double getMaxEig(torch::Tensor TMat, double *U){
    //soving eigenvalues problem
    std::tuple<torch::Tensor, torch::Tensor> result = torch::linalg::eig(TMat);
    //getting maximum eigenvalue and eigenvector
    double max_eig_val = std::numeric_limits<double>::min();
    torch::Tensor max_eig_vec = torch::zeros({4}, torch::TensorOptions().dtype(torch::kDouble));
    auto q = max_eig_vec.accessor<double, 1>();
    torch::Tensor eig_vals = torch::zeros({4, 2}, torch::TensorOptions().dtype(torch::kDouble));
    eig_vals.copy_(std::get<0>(result));
    torch::Tensor eig_vecs = torch::zeros({4, 4}, torch::TensorOptions().dtype(torch::kDouble));
    eig_vecs.copy_(std::get<1>(result));
    auto eig_val = eig_vals.accessor<double, 2>();

    for(int i=0; i<4; i++){    
        if(max_eig_val < eig_val[i][0]){
            max_eig_val = eig_val[i][0];        
            for(int j=0;j<4; j++){
                max_eig_vec[j] = eig_vecs[j][i];
            }
        }
    }
    
    // rotation matrix
    U[0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
    U[1] = 2.0*(q[1]*q[2] - q[0]*q[3]);
    U[2] = 2.0*(q[1]*q[3] + q[0]*q[2]);

    U[3] = 2.0*(q[1]*q[2] + q[0]*q[3]);
    U[4] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
    U[5] = 2.0*(q[2]*q[3] - q[0]*q[1]);

    U[6] = 2.0*(q[1]*q[3] - q[0]*q[2]);
    U[7] = 2.0*(q[2]*q[3] + q[0]*q[1]);
    U[8] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];

    return max_eig_val;
    
}


void Coords2RMSDGPU_forward(   torch::Tensor centered_coords_src, 
                                torch::Tensor centered_coords_dst, 
                                torch::Tensor output, 
                                torch::Tensor num_atoms,
                                torch::Tensor UT
                        ){
    CHECK_GPU_INPUT(centered_coords_src);
    CHECK_GPU_INPUT(centered_coords_dst);
    CHECK_GPU_INPUT(output);
    CHECK_GPU_INPUT(UT);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(centered_coords_src.ndimension() != 2 || centered_coords_dst.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    
    int batch_size = num_atoms.size(0);
    int atoms_stride = centered_coords_src.size(1)/3;
    torch::Tensor T = torch::zeros({batch_size, 4, 4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kDouble));
        
    AT_DISPATCH_FLOATING_TYPES(centered_coords_src.type(), "Coords2RMSD_GPU_forward", ([&]{
        //correlation matrix T
        gpu_correlationMatrix<scalar_t>(centered_coords_src.data<scalar_t>(),
                                        centered_coords_dst.data<scalar_t>(),
                                        T.data<double>(),
                                        num_atoms.data<int>(), batch_size, atoms_stride);
    }));

    //computing eigenvectors and eigenvalues on CPU: CPU is not implemented in Torch, need Magma!!
    torch::Tensor cpu_T = T.to(torch::TensorOptions().device(torch::kCPU));
    for(int i=0;i<batch_size;i++){
        torch::Tensor cpu_T_single = cpu_T[i];
        torch::Tensor UT_single = UT[i];
        torch::Tensor centered_coords_src_single = centered_coords_src[i];
        torch::Tensor centered_coords_dst_single = centered_coords_dst[i];
        
        double U[9];
        double max_eig_val = getMaxEig(cpu_T_single, U);
        
        for(int j=0;j<3;j++){
            for(int k=0;k<3;k++){
                UT_single[j][k] = U[3*k+j];
        }}
        
        int num_atoms_cpu = num_atoms[i].item().toInt();
        
        //computing R2 coefficient
        torch::Tensor R2 = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kDouble));
        torch::Tensor R2_tmp = torch::zeros({3}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kDouble));
        
        AT_DISPATCH_FLOATING_TYPES(centered_coords_src.type(), "Coords2RMSD_GPU_forward", ([&]{
            gpu_computeR2<scalar_t>(centered_coords_src_single.data<scalar_t>(), num_atoms_cpu, R2_tmp.data<double>());
            R2 += R2_tmp.sum();
            gpu_computeR2<scalar_t>(centered_coords_dst_single.data<scalar_t>(), num_atoms_cpu, R2_tmp.data<double>());
            R2 += R2_tmp.sum();
        }));

        double R2_cpu = R2[0].item().toDouble();
        output[i] = sqrt((R2_cpu - 2.0*fabs(max_eig_val) + EPS_LIMIT)/(double(num_atoms_cpu)));
    }
}

void Coords2RMSD_forward(   torch::Tensor centered_coords_src, 
                            torch::Tensor centered_coords_dst, 
                            torch::Tensor output, 
                            torch::Tensor num_atoms,
                            torch::Tensor UT
                        ){
    CHECK_CPU_INPUT(centered_coords_src);
    CHECK_CPU_INPUT(centered_coords_dst);
    CHECK_CPU_INPUT(output);
    CHECK_CPU_INPUT(UT);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(centered_coords_src.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    int batch_size = num_atoms.size(0);
    for(int i=0; i<batch_size; i++){
        torch::Tensor UT_single = UT[i];
        torch::Tensor centered_coords_src_single = centered_coords_src[i];
        torch::Tensor centered_coords_dst_single = centered_coords_dst[i];
        int single_num_atoms = num_atoms[i].item().toInt();
        //correlation matrix R
        cMatrix33<double> R;
        AT_DISPATCH_FLOATING_TYPES(centered_coords_src.type(), "Coords2RMSD_forward", ([&]{    
            for(int j=0; j<single_num_atoms; j++){
                cVector3<scalar_t> ce_src_atom(centered_coords_src_single.data<scalar_t>()+3*j);
                cVector3<scalar_t> ce_dst_atom(centered_coords_dst_single.data<scalar_t>()+3*j);
                R.m[0][0] += ((double)ce_src_atom.v[0])*((double)ce_dst_atom.v[0]);
                R.m[0][1] += ((double)ce_src_atom.v[0])*((double)ce_dst_atom.v[1]);
                R.m[0][2] += ((double)ce_src_atom.v[0])*((double)ce_dst_atom.v[2]);
                R.m[1][0] += ((double)ce_src_atom.v[1])*((double)ce_dst_atom.v[0]);
                R.m[1][1] += ((double)ce_src_atom.v[1])*((double)ce_dst_atom.v[1]);
                R.m[1][2] += ((double)ce_src_atom.v[1])*((double)ce_dst_atom.v[2]);
                R.m[2][0] += ((double)ce_src_atom.v[2])*((double)ce_dst_atom.v[0]);
                R.m[2][1] += ((double)ce_src_atom.v[2])*((double)ce_dst_atom.v[1]);
                R.m[2][2] += ((double)ce_src_atom.v[2])*((double)ce_dst_atom.v[2]);
            }
        }));

        //Converting R to the T matrix
        at::Tensor Fmat = torch::zeros({4,4}, torch::TensorOptions().dtype(torch::kDouble));
        double *F = Fmat.data<double>();
        F[0] = R.m[0][0]+R.m[1][1]+R.m[2][2];F[1] = R.m[1][2]-R.m[2][1];F[2] = R.m[2][0]-R.m[0][2];F[3] = R.m[0][1]-R.m[1][0];
        F[4] = R.m[1][2]-R.m[2][1];F[5] = R.m[0][0]-R.m[1][1]-R.m[2][2];F[6] = R.m[0][1]+R.m[1][0];F[7] = R.m[0][2]+R.m[2][0];
        F[8] = R.m[2][0]-R.m[0][2];F[9] = R.m[0][1]+R.m[1][0];F[10] = -R.m[0][0]+R.m[1][1]-R.m[2][2];F[11] = R.m[1][2]+R.m[2][1];
        F[12] = R.m[0][1]-R.m[1][0];F[13] = R.m[0][2]+R.m[2][0];F[14] = R.m[1][2]+R.m[2][1];F[15] = -R.m[0][0]-R.m[1][1]+R.m[2][2];
            
        double U[9];
        double max_eig_val = getMaxEig(Fmat, U);
                
        for(int j=0;j<3;j++){
            for(int k=0;k<3;k++){
                UT_single[j][k] = U[3*k+j];
        }}
        

        double R2 = 0.0;
        AT_DISPATCH_FLOATING_TYPES(centered_coords_src.type(), "Coords2RMSD_forward", ([&]{ 
            
            //computing R2 coefficient
            for(int j=0; j<single_num_atoms; j++){
                cVector3<scalar_t> ce_src_atom(centered_coords_src_single.data<scalar_t>()+3*j);
                cVector3<scalar_t> ce_dst_atom(centered_coords_dst_single.data<scalar_t>()+3*j);
                cVector3<double> src_at_double((double)ce_src_atom.v[0], (double)ce_src_atom.v[1], (double)ce_src_atom.v[2]);
                cVector3<double> dst_at_double((double)ce_dst_atom.v[0], (double)ce_dst_atom.v[1], (double)ce_dst_atom.v[2]);
                R2 += src_at_double.norm2() + dst_at_double.norm2();
            }
        }));
        
        output[i] = sqrt((R2 - 2.0*fabs(max_eig_val) + EPS_LIMIT)/(double(single_num_atoms)));
    }
}