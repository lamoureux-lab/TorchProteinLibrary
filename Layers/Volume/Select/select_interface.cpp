#include "select_interface.h"
#include <iostream>
#include <string>
#include <Kernels.h>


void SelectVolume_forward(  at::Tensor volume,
                            at::Tensor coords,
                            at::Tensor num_atoms,
                            at::Tensor features,
                            float res
                        ){
    if( volume.dtype() != at::kFloat || coords.dtype() != at::kFloat || num_atoms.dtype() != at::kInt 
    || features.dtype() != at::kFloat){
        throw("Incorrect tensor types");
    }
    if( (!volume.type().is_cuda()) || (!coords.type().is_cuda()) || (!num_atoms.type().is_cuda()) 
    || (!features.type().is_cuda())){
        throw("Incorrect device");
    }
    if(coords.ndimension() != 2){
        throw("Incorrect input ndim");
    }


    int batch_size = coords.size(0);
    int num_features = features.size(1);
    int spatial_dim = volume.size(2);
    int max_num_atoms = coords.size(1)/3;

    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_volume = volume[i];
        at::Tensor single_coords = coords[i];
        at::Tensor single_features = features[i];

        int single_num_atoms = at::Scalar(num_atoms[i]).toInt();
        gpu_selectFromTensor(   single_features.data<float>(), num_features,
                                single_volume.data<float>(), spatial_dim,
                                single_coords.data<float>(), single_num_atoms, max_num_atoms, res);
    }
}                        

