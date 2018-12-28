#include "coords2rmsd_interface.h"
#include "../RMSDKernels.h"
#include <iostream>
#include <tuple>
#include <limits>


void Coords2RMSD_GPU_forward(   at::Tensor re_coordinates_src, at::Tensor re_coordinates_dst, 
                            at::Tensor output, at::Tensor num_atoms,
                            at::Tensor Ut_coordinates_dst
                        ){
    // if( re_coordinates_src.dtype() != at::kDouble || re_coordinates_dst.dtype() != at::kDouble || num_atoms.dtype() != at::kInt 
    // || output.dtype() != at::kDouble || Ut_coordinates_dst.dtype() != at::kDouble){
    //     throw("Incorrect tensor types");
    // }
    if( (!re_coordinates_src.type().is_cuda()) || (!re_coordinates_dst.type().is_cuda()) || (!num_atoms.type().is_cuda()) 
        || (!output.type().is_cuda()) || (!Ut_coordinates_dst.type().is_cuda()) ){
        throw("Incorrect device");
    }
    if(re_coordinates_src.ndimension() != 2){
        throw("Incorrect input ndim");
    }
    
    int batch_size = num_atoms.size(0);
    // at::Tensor T = at::CUDA(at::kDouble).zeros({batch_size, 4, 4});
    // at::Tensor rot_mat_t = at::CUDA(at::kDouble).zeros({batch_size, 3, 3});
    at::Tensor T = torch::zeros({batch_size, 4, 4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kDouble));
    at::Tensor rot_mat_t = torch::zeros({batch_size, 3, 3}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kDouble));
    //correlation matrix T
    cpu_correlationMatrix(  re_coordinates_src.data<double>(),
                            re_coordinates_dst.data<double>(),
                            T.data<double>(),
                            num_atoms.data<int>(), batch_size, re_coordinates_src.size(1));

    //computing eigenvectors and eigenvalues on CPU: CPU is not implemented in Torch, need Magma!!
    at::Tensor cpu_T = T.toBackend(at::Backend::CUDA);
    for(int i=0;i<batch_size;i++){
        at::Tensor cpu_T_single = cpu_T[i];
        at::Tensor rot_mat_t_single = rot_mat_t[i];
        at::Tensor re_coords_src_single = re_coordinates_src[i];
        at::Tensor re_coords_dst_single = re_coordinates_dst[i];
        

        //soving eigenvalues problem
        std::tuple<at::Tensor, at::Tensor> result = cpu_T_single.eig(true);
        
        //getting maximum eigenvalue and eigenvector
        double max_eig_val = std::numeric_limits<double>::min();
        // at::Tensor max_eig_vec = at::CPU(at::kDouble).zeros({4});
        at::Tensor max_eig_vec = torch::zeros({4}, torch::TensorOptions().dtype(torch::kDouble));
        auto q = max_eig_vec.accessor<double, 1>();

        // at::Tensor eig_vals = at::CPU(at::kDouble).zeros({4,2});
        at::Tensor eig_vals = torch::zeros({4, 2}, torch::TensorOptions().dtype(torch::kDouble));
        eig_vals.copy_(std::get<0>(result));
        // at::Tensor eig_vecs = at::CPU(at::kDouble).zeros({4,4}); 
        at::Tensor eig_vecs = torch::zeros({4, 4}, torch::TensorOptions().dtype(torch::kDouble));
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
        double U[9];
        U[0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
        U[1] = 2.0*(q[1]*q[2] - q[0]*q[3]);
        U[2] = 2.0*(q[1]*q[3] + q[0]*q[2]);

        U[3] = 2.0*(q[1]*q[2] + q[0]*q[3]);
        U[4] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
        U[5] = 2.0*(q[2]*q[3] - q[0]*q[1]);

        U[6] = 2.0*(q[1]*q[3] - q[0]*q[2]);
        U[7] = 2.0*(q[2]*q[3] + q[0]*q[1]);
        U[8] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
        
        for(int j=0;j<3;j++){
            for(int k=0;k<3;k++){
                rot_mat_t_single[j][k] = U[3*k+j];
        }}
        
        // int num_atoms_cpu = at::Scalar(num_atoms[i]).toInt();
        int num_atoms_cpu = num_atoms[i].item().toInt();
        //computing R2 coefficient
        // at::Tensor R2 = at::CUDA(at::kDouble).zeros({1});
        at::Tensor R2 = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kDouble));
        // at::Tensor R2_tmp = at::CUDA(at::kDouble).zeros({3});
        at::Tensor R2_tmp = torch::zeros({3}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kDouble));
        cpu_computeR2(re_coords_src_single.data<double>(), num_atoms_cpu, R2_tmp.data<double>());
        R2 += R2_tmp.sum();
        cpu_computeR2(re_coords_dst_single.data<double>(), num_atoms_cpu, R2_tmp.data<double>());
        R2 += R2_tmp.sum();
    
        // double R2_cpu = at::Scalar(R2[0]).toDouble();       
        double R2_cpu = R2[0].item().toDouble();
        double rmsd = (R2_cpu - 2.0*fabs(max_eig_val))/(double(num_atoms_cpu));
        output[i] = rmsd;
    }

    cpu_transformCoordinates(   re_coordinates_dst.data<double>(),
                                Ut_coordinates_dst.data<double>(),
                                rot_mat_t.data<double>(),
                                batch_size, re_coordinates_dst.size(1));
}
