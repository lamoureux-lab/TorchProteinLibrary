#include "coords2eps_interface.h"
#include <iostream>
#include "nUtil.h"
#include "Kernels.h"

void Coords2Eps_forward(torch::Tensor coords, torch::Tensor assigned_params, torch::Tensor num_atoms, torch::Tensor eps, float resolution){
    CHECK_GPU_INPUT_TYPE(coords, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(assigned_params, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(eps, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    
    gpu_computeCoords2Volume(   coords.data<float>(), 
                                assigned_params.data<float>(),
                                num_atoms.data<int>(),
                                eps.data<float>(),
                                eps.size(0), //batch_size
                                eps.size(1), //box_size
                                coords.size(1), //coords_stride
                                resolution);
}

void Coords2Eps_backward(   torch::Tensor gradOutput, torch::Tensor gradInput, 
                            torch::Tensor coords, torch::Tensor assigned_params, torch::Tensor num_atoms,
                            float resolution){
    CHECK_GPU_INPUT_TYPE(gradOutput, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(gradInput, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(coords, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(assigned_params, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    
}