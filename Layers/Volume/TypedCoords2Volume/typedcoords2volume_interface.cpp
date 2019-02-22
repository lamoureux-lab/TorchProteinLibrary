#include "typedcoords2volume_interface.h"
#include <iostream>
#include <string>
#include <Kernels.h>

void TypedCoords2Volume_forward(    at::Tensor input_coords,
                                    at::Tensor volume,
                                    at::Tensor num_atoms_of_type,
                                    at::Tensor offsets,
                                    float resolution,
                                    int mode){
    int num_atom_types=11;
    // if( (!input_coords.type().is_cuda()) || (!volume.type().is_cuda()) || (!num_atoms_of_type.type().is_cuda()) 
    //     || (!offsets.type().is_cuda())
    //     || input_coords.dtype() != at::kDouble || volume.dtype() != at::kFloat
    //     || num_atoms_of_type.dtype() != at::kInt || offsets.dtype() != at::kInt){
    //         std::cout<<"Incorrect tensor types"<<std::endl;
    //         throw("Incorrect tensor types");

    // }
    if(input_coords.ndimension() != 2){
        std::cout<<"Incorrect input ndim"<<std::endl;
        throw("Incorrect input ndim");
    }
    if(mode!=1 && mode!=2){
        std::cout<<"Incorrect mode"<<std::endl;
        throw("Incorrect mode");
    }
    int batch_size = input_coords.size(0);

    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_num_atoms_of_type = num_atoms_of_type[i];
        at::Tensor single_offsets = offsets[i];
        at::Tensor single_volume = volume[i];
        at::Tensor single_input_coords = input_coords[i];
        
        gpu_computeCoords2Volume(   single_input_coords.data<double>(), 
                                    single_num_atoms_of_type.data<int>(), 
                                    single_offsets.data<int>(), 
                                    single_volume.data<float>(), single_volume.size(1), num_atom_types, resolution, mode);
    }
    
}
void TypedCoords2Volume_backward(   at::Tensor grad_volume,
                                    at::Tensor grad_coords,
                                    at::Tensor coords,
                                    at::Tensor num_atoms_of_type,
                                    at::Tensor offsets,
                                    float resolution,
                                    int mode){
    int num_atom_types=11;
    // if( (!grad_coords.type().is_cuda()) || (!grad_volume.type().is_cuda()) || (!num_atoms_of_type.type().is_cuda()) 
    //     || (!offsets.type().is_cuda()) || (!coords.type().is_cuda())
    //     || grad_coords.dtype() != at::kDouble || coords.dtype() != at::kDouble || grad_volume.dtype() != at::kFloat
    //     || num_atoms_of_type.dtype() != at::kInt || offsets.dtype() != at::kInt){
    //         throw("Incorrect tensor types");
    // }
    if(grad_coords.ndimension() != 2){
        throw("Incorrect input ndim");
    }
    if(mode!=1 && mode!=2){
        std::cout<<"Incorrect mode"<<std::endl;
        throw("Incorrect mode");
    }
    int batch_size = grad_coords.size(0);
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_num_atoms_of_type = num_atoms_of_type[i];
        at::Tensor single_offsets = offsets[i];
        at::Tensor single_grad_volume = grad_volume[i];
        at::Tensor single_coords = coords[i];
        at::Tensor single_grad_coords = grad_coords[i];
        
        
        gpu_computeVolume2Coords(   single_coords.data<double>(), 
                                    single_grad_coords.data<double>(),
                                    single_num_atoms_of_type.data<int>(),
                                    single_offsets.data<int>(), 
                                    single_grad_volume.data<float>(), 
                                    single_grad_volume.size(1), num_atom_types, resolution, mode);
    }
    
}
