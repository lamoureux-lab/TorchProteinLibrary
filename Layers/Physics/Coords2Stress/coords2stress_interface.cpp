#include "coords2elec_interface.h"
#include <iostream>
#include "nUtil.h"
#include "Kernels.h"

void Coords2Stress_forward( torch::Tensor coords, torch::Tensor gamma, torch::Tensor num_atoms, torch::Tensor stress, 
                            float resolution){
    CHECK_GPU_INPUT_TYPE(coords, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(gamma, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(stress, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    
    int batch_size = num_atoms.size(0);
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
    }
}