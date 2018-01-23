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
        if(src->nDimension == 1){
            cRMSD crmsd( THDoubleTensor_data(ce_src), THDoubleTensor_data(ce_dst), 
                        THDoubleTensor_data(U_ce_src), THDoubleTensor_data(UT_ce_dst), 
                        THIntTensor_get1d(num_atoms, 0));
            THDoubleTensor_set1d(rmsd, 0, crmsd.compute(THDoubleTensor_data(src), THDoubleTensor_data(dst)));
        }else if(src->nDimension == 2){
            std::cout<<"Not implemented\n";
        }
    }
    void Coords2RMSD_backward(THDoubleTensor *grad_atoms, THDoubleTensor *grad_output,
                                THDoubleTensor *ce_src, THDoubleTensor *ce_dst,
                                THDoubleTensor *U_ce_src, THDoubleTensor *UT_ce_dst,
                                THIntTensor *num_atoms
                            ){
        if(grad_atoms->nDimension == 1){
            cRMSD rmsd( THDoubleTensor_data(ce_src), THDoubleTensor_data(ce_dst), 
                        THDoubleTensor_data(U_ce_src), THDoubleTensor_data(UT_ce_dst), 
                        THIntTensor_get1d(num_atoms, 0));
            return rmsd.grad(THDoubleTensor_data(grad_atoms), THDoubleTensor_data(grad_output));
        }else{
            std::cout<<"Not implemented\n";
        }
    }
}