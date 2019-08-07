#include "coords2rmsd_interface.h"
#include <cRMSD.h>
#include <iostream>
#include <string>
#include <nUtil.h>


void Coords2RMSD_CPU_forward(   torch::Tensor src, torch::Tensor dst, torch::Tensor rmsd,
                            torch::Tensor ce_src, torch::Tensor ce_dst,
                            torch::Tensor U_ce_src, torch::Tensor UT_ce_dst,
                            torch::Tensor num_atoms
                        ){
    CHECK_CPU_INPUT(src);
    CHECK_CPU_INPUT(dst);
    CHECK_CPU_INPUT(rmsd);
    CHECK_CPU_INPUT(ce_src);
    CHECK_CPU_INPUT(ce_dst);
    CHECK_CPU_INPUT(U_ce_src);
    CHECK_CPU_INPUT(UT_ce_dst);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(src.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    
    int batch_size = src.size(0);
    auto num_atoms_acc = num_atoms.accessor<int,1>();
    // #pragma omp parallel for num_threads(10)
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_src = src[i];
        torch::Tensor single_dst = dst[i];
        torch::Tensor single_ce_src = ce_src[i];
        torch::Tensor single_ce_dst = ce_dst[i];
        torch::Tensor single_U_ce_src = U_ce_src[i];
        torch::Tensor single_UT_ce_dst = UT_ce_dst[i];

        cRMSD<double> crmsd( single_ce_src.data<double>(), single_ce_dst.data<double>(), 
                            single_U_ce_src.data<double>(), single_UT_ce_dst.data<double>(), 
                            num_atoms_acc[i]);
        rmsd[i] = crmsd.compute(single_src.data<double>(), single_dst.data<double>());
    }
}
void Coords2RMSD_CPU_backward(  torch::Tensor grad_atoms, torch::Tensor grad_output,
                            torch::Tensor ce_src, torch::Tensor ce_dst,
                            torch::Tensor U_ce_src, torch::Tensor UT_ce_dst,
                            torch::Tensor num_atoms
                        ){
    CHECK_CPU_INPUT(grad_atoms);
    CHECK_CPU_INPUT(grad_output);
    CHECK_CPU_INPUT(ce_src);
    CHECK_CPU_INPUT(ce_dst);
    CHECK_CPU_INPUT(U_ce_src);
    CHECK_CPU_INPUT(UT_ce_dst);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(grad_atoms.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    int batch_size = grad_atoms.size(0);
    auto grad_output_acc = grad_output.accessor<double, 1>();
    auto num_atoms_acc = num_atoms.accessor<int,1>();

    // #pragma omp parallel for num_threads(10)
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_grad_atoms = grad_atoms[i];
        torch::Tensor single_ce_src = ce_src[i];
        torch::Tensor single_ce_dst = ce_dst[i];
        torch::Tensor single_U_ce_src = U_ce_src[i];
        torch::Tensor single_UT_ce_dst = UT_ce_dst[i];
        
        double single_grad_output = grad_output_acc[i];
        
        cRMSD<double> rmsd( single_ce_src.data<double>(), single_ce_dst.data<double>(), 
                    single_U_ce_src.data<double>(), single_UT_ce_dst.data<double>(), 
                    num_atoms_acc[i]);

        rmsd.grad(single_grad_atoms.data<double>(), &single_grad_output);
    }
}
