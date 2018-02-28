#include <TH/TH.h>
#include "cRMSD.h"
#include <iostream>
#include <string>

extern "C" {
    void Coords2RMSD_forward( THDoubleTensor *src, THDoubleTensor *dst, THDoubleTensor *rmsd,
                                THDoubleTensor *ce_src, THDoubleTensor *ce_dst,
                                THDoubleTensor *U_ce_src, THDoubleTensor *UT_ce_dst,
                                THIntTensor *num_atoms
                            ){
        if(src->nDimension == 2){
            int batch_size = src->size[0];

            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THDoubleTensor *single_src = THDoubleTensor_new();
                THDoubleTensor *single_dst = THDoubleTensor_new();
                THDoubleTensor *single_ce_src = THDoubleTensor_new();
                THDoubleTensor *single_ce_dst = THDoubleTensor_new();
                THDoubleTensor *single_U_ce_src = THDoubleTensor_new();
                THDoubleTensor *single_UT_ce_dst = THDoubleTensor_new();
                
                THDoubleTensor_select(single_src, src, 0, i);
                THDoubleTensor_select(single_dst, dst, 0, i);
                THDoubleTensor_select(single_ce_src, ce_src, 0, i);
                THDoubleTensor_select(single_ce_dst, ce_dst, 0, i);
                THDoubleTensor_select(single_U_ce_src, U_ce_src, 0, i);
                THDoubleTensor_select(single_UT_ce_dst, UT_ce_dst, 0, i);

                cRMSD crmsd( THDoubleTensor_data(single_ce_src), THDoubleTensor_data(single_ce_dst), 
                            THDoubleTensor_data(single_U_ce_src), THDoubleTensor_data(single_UT_ce_dst), 
                            THIntTensor_get1d(num_atoms, i));
                THDoubleTensor_set1d(rmsd, i, crmsd.compute(THDoubleTensor_data(single_src), THDoubleTensor_data(single_dst)));

                THDoubleTensor_free(single_src);
                THDoubleTensor_free(single_dst);
                THDoubleTensor_free(single_ce_src);
                THDoubleTensor_free(single_ce_dst);
                THDoubleTensor_free(single_U_ce_src);
                THDoubleTensor_free(single_UT_ce_dst);
            }
        }else{
            std::cout<<"Not implemented\n";
        }
    }
    void Coords2RMSD_backward(THDoubleTensor *grad_atoms, THDoubleTensor *grad_output,
                                THDoubleTensor *ce_src, THDoubleTensor *ce_dst,
                                THDoubleTensor *U_ce_src, THDoubleTensor *UT_ce_dst,
                                THIntTensor *num_atoms
                            ){
        if(grad_atoms->nDimension == 2){
            int batch_size = grad_atoms->size[0];
            
            // #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                
                THDoubleTensor *single_grad_atoms = THDoubleTensor_new();
                // THDoubleTensor *single_grad_output = THDoubleTensor_new();
                THDoubleTensor *single_ce_src = THDoubleTensor_new();
                THDoubleTensor *single_ce_dst = THDoubleTensor_new();
                THDoubleTensor *single_U_ce_src = THDoubleTensor_new();
                THDoubleTensor *single_UT_ce_dst = THDoubleTensor_new();
                
                std::cout<<'a'<<std::endl;
                
                THDoubleTensor_select(single_grad_atoms, grad_atoms, 0, i);
                
                // THDoubleTensor_select(single_grad_output, grad_output, 0, i);
                double single_grad_output = THDoubleTensor_get1d(grad_output, i);
                std::cout<<'b'<<std::endl;
                THDoubleTensor_select(single_ce_src, ce_src, 0, i);
                THDoubleTensor_select(single_ce_dst, ce_dst, 0, i);
                THDoubleTensor_select(single_U_ce_src, U_ce_src, 0, i);
                THDoubleTensor_select(single_UT_ce_dst, UT_ce_dst, 0, i);
                

                cRMSD rmsd( THDoubleTensor_data(single_ce_src), THDoubleTensor_data(single_ce_dst), 
                            THDoubleTensor_data(single_U_ce_src), THDoubleTensor_data(single_UT_ce_dst), 
                            THIntTensor_get1d(num_atoms, i));
                rmsd.grad(THDoubleTensor_data(single_grad_atoms), &single_grad_output);
                
                THDoubleTensor_free(single_grad_atoms);
                THDoubleTensor_free(single_ce_src);
                THDoubleTensor_free(single_ce_dst);
                THDoubleTensor_free(single_U_ce_src);
                THDoubleTensor_free(single_UT_ce_dst);
            }
        }else{
            std::cout<<"Not implemented\n";
        }
    }
}