#include "select_interface.h"
#include <iostream>
#include <string>
#include <Kernels.h>
#include <nUtil.h>

void SelectVolume_forward(  torch::Tensor volume,
                            torch::Tensor coords,
                            torch::Tensor num_atoms,
                            torch::Tensor features,
                            float res
                        ){
    CHECK_GPU_INPUT(volume);
    CHECK_GPU_INPUT(coords);
    CHECK_GPU_INPUT(features);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }


    int batch_size = coords.size(0);
    int num_features = features.size(1);
    int spatial_dim = volume.size(2);
    int max_num_atoms = coords.size(1)/3;

    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_volume = volume[i];
        torch::Tensor single_coords = coords[i];
        torch::Tensor single_features = features[i];

        int single_num_atoms = num_atoms[i].item().toInt();
        gpu_selectFromTensor(   single_features.data<float>(), num_features,
                                single_volume.data<float>(), spatial_dim,
                                single_coords.data<float>(), single_num_atoms, max_num_atoms, res);
    }
}                        

