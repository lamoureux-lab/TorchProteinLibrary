#include "coords2elec_interface.h"
#include <iostream>
#include "nUtil.h"
#include "KernelsStress.h"

void Coords2Stress_forward( torch::Tensor coords, torch::Tensor vectors, torch::Tensor num_atoms, torch::Tensor volume, 
                            float resolution){
    CHECK_GPU_INPUT_TYPE(coords, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(vectors, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(volume, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    
    int batch_size = num_atoms.size(0);
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_coords = coords[i];
        torch::Tensor single_vectors = vectors[i];
        torch::Tensor single_volume = volume[i];
        gpu_projectToGrid(  single_coords.data<float>(),
                            single_vectors.data<float>(),
                            num_atoms[i].item().toInt(),
                            single_volume.data<float>(),
                            single_volume.size(1),//box_size
                            resolution);
    }
}