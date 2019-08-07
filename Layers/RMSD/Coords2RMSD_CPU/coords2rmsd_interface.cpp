#include "coords2rmsd_interface.h"
#include <cRMSD.h>
#include <iostream>
#include <string>


void Coords2RMSD_CPU_forward(   torch::Tensor src, torch::Tensor dst, torch::Tensor rmsd,
                            torch::Tensor ce_src, torch::Tensor ce_dst,
                            torch::Tensor U_ce_src, torch::Tensor UT_ce_dst,
                            torch::Tensor num_atoms
                        ){
    // if( src.dtype() != torch::kDouble || dst.dtype() != torch::kDouble || rmsd.dtype() != torch::kDouble
    // || ce_src.dtype() != torch::kDouble || ce_dst.dtype() != torch::kDouble || U_ce_src.dtype() != torch::kDouble
    // || UT_ce_dst.dtype() != torch::kDouble || num_atoms.dtype() != torch::kInt){
    //     throw("Incorrect tensor types");
    // }
    if( (src.type().is_cuda()) || (dst.type().is_cuda()) || (rmsd.type().is_cuda())
        || (ce_src.type().is_cuda()) || (ce_dst.type().is_cuda()) || (U_ce_src.type().is_cuda()) 
        || (UT_ce_dst.type().is_cuda()) || (num_atoms.type().is_cuda()) ){
        throw("Incorrect device");
    }
    if(src.ndimension() != 2){
        throw("Incorrect input ndim");
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
    // if( grad_atoms.dtype() != torch::kDouble || grad_output.dtype() != torch::kDouble
    // || ce_src.dtype() != torch::kDouble || ce_dst.dtype() != torch::kDouble || U_ce_src.dtype() != torch::kDouble
    // || UT_ce_dst.dtype() != torch::kDouble || num_atoms.dtype() != torch::kInt){
    //     std::cout<<"Incorrect tensor types"<<std::endl;
    //     throw("Incorrect tensor types");
    // }
    if( (grad_atoms.type().is_cuda()) || (grad_output.type().is_cuda())
        || (ce_src.type().is_cuda()) || (ce_dst.type().is_cuda()) || (U_ce_src.type().is_cuda()) 
        || (UT_ce_dst.type().is_cuda()) || (num_atoms.type().is_cuda()) ){
        std::cout<<"Incorrect device"<<std::endl;
        throw("Incorrect device");
    }
    if(grad_atoms.ndimension() != 2){
        std::cout<<"Incorrect input ndim"<<std::endl;
        throw("Incorrect input ndim");
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
