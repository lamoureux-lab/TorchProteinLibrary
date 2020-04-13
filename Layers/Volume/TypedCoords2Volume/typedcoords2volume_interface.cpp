#include "typedcoords2volume_interface.h"
#include <iostream>
#include <string>
#include <Kernels.h>
#include <nUtil.h>

void TypedCoords2Volume_forward(    torch::Tensor input_coords,
                                    torch::Tensor volume,
                                    torch::Tensor num_atoms,
                                    float resolution){
    CHECK_GPU_INPUT(volume);
    CHECK_GPU_INPUT(input_coords);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(input_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    int batch_size = input_coords.size(0);
    auto a_num_atoms = num_atoms.accessor<int, 1>();
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_volume = volume[i];
        torch::Tensor single_input_coords = input_coords[i];
        AT_DISPATCH_FLOATING_TYPES(input_coords.type(), "TypedCoords2Volume_forward", ([&]{
            gpu_computeCoords2Volume<scalar_t>( single_input_coords.data<scalar_t>(), 
                                                a_num_atoms[i], 
                                                single_volume.data<scalar_t>(), 
                                                single_volume.size(1), resolution);
        }));
    }
    
}
void TypedCoords2Volume_backward(   torch::Tensor grad_volume,
                                    torch::Tensor grad_coords,
                                    torch::Tensor coords,
                                    torch::Tensor num_atoms,
                                    float resolution){
    CHECK_GPU_INPUT(grad_volume);
    CHECK_GPU_INPUT(grad_coords);
    CHECK_GPU_INPUT(coords);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(grad_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    int batch_size = grad_coords.size(0);
    auto a_num_atoms = num_atoms.accessor<int, 1>();
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_grad_volume = grad_volume[i];
        torch::Tensor single_coords = coords[i];
        torch::Tensor single_grad_coords = grad_coords[i];
        
        AT_DISPATCH_FLOATING_TYPES(grad_coords.type(), "TypedCoords2Volume_backward", ([&]{
            gpu_computeVolume2Coords<scalar_t>(single_coords.data<scalar_t>(), 
                                        single_grad_coords.data<scalar_t>(),
                                        a_num_atoms[i],
                                        single_grad_volume.data<scalar_t>(), 
                                        single_grad_volume.size(1), resolution);
        }));
    }
    
}
