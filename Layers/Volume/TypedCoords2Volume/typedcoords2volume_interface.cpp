#include "typedcoords2volume_interface.h"
#include <iostream>
#include <string>
#include <HashKernel.h>
#include <nUtil.h>

//#define _OPENMP
//#include <ATen/ParallelOpenMP.h>

void TypedCoords2Volume_forward(    torch::Tensor input_coords,
                                    torch::Tensor volume,
                                    torch::Tensor num_atoms,
                                    float resolution, int num_neighbours,
                                    torch::Tensor gridParticleHash,
                                    torch::Tensor gridParticleIndex,
                                    torch::Tensor cellStart,
                                    torch::Tensor cellStop,
                                    torch::Tensor sortedPos){
    CHECK_GPU_INPUT(volume);
    CHECK_GPU_INPUT(input_coords);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);

    CHECK_GPU_INPUT_TYPE(gridParticleHash, torch::kInt64);
    CHECK_GPU_INPUT_TYPE(gridParticleIndex, torch::kInt64);
    CHECK_GPU_INPUT_TYPE(cellStart, torch::kInt64);
    CHECK_GPU_INPUT_TYPE(cellStop, torch::kInt64);
    CHECK_GPU_INPUT(sortedPos);

    if(input_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    int batch_size = input_coords.size(0);
    auto a_num_atoms = num_atoms.accessor<int, 1>();
    // #pragma omp parallel for
    // for(int i=0; i<batch_size; i++){
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++)
    {
        torch::Tensor single_volume = volume[i];
        torch::Tensor single_input_coords = input_coords[i];
        torch::Tensor single_particleHash = gridParticleHash[i];
        torch::Tensor single_particleIndex = gridParticleIndex[i];
        torch::Tensor single_cellStart = cellStart[i];
        torch::Tensor single_cellStop = cellStop[i];
        torch::Tensor single_sortedPos = sortedPos[i];
        int device = single_input_coords.device().index();
        AT_DISPATCH_FLOATING_TYPES(input_coords.type(), "TypedCoords2Volume_forward", ([&]{
            gpu_computeCoords2Volume<scalar_t>( single_input_coords.data<scalar_t>(), 
                                                a_num_atoms[i], 
                                                single_volume.data<scalar_t>(), 
                                                single_volume.size(1), resolution, num_neighbours,
                                                (single_particleHash[0]).data<long>(),
                                                (single_particleHash[1]).data<long>(),
                                                (single_particleIndex[0]).data<long>(),
                                                (single_particleIndex[1]).data<long>(),
                                                single_cellStart.data<long>(),
                                                single_cellStop.data<long>(),
                                                single_sortedPos.data<scalar_t>(),
                                                device);
        }));
    }});
    
}
void TypedCoords2Volume_backward(   torch::Tensor grad_volume,
                                    torch::Tensor grad_coords,
                                    torch::Tensor coords,
                                    torch::Tensor num_atoms,
                                    float resolution,
                                    int num_neighbours){
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
                                        single_grad_volume.size(1), resolution, num_neighbours);
        }));
    }
    
}


