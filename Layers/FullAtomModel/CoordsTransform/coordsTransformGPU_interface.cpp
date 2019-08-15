#include "cConformation.h"
#include <iostream>
#include <string>
#include "nUtil.h"
#include "coordsTransform_interface.h"
#include <TransformCUDAKernels.h>


void CoordsTranslateGPU_forward(    torch::Tensor input_coords, 
                                    torch::Tensor output_coords,
                                    torch::Tensor T,
                                    torch::Tensor num_atoms
                                ){
    CHECK_GPU_INPUT(input_coords);
    CHECK_GPU_INPUT(output_coords);
    CHECK_GPU_INPUT(T);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(input_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    int batch_size = input_coords.size(0);
    int atoms_stride = input_coords.size(1)/3;
    AT_DISPATCH_FLOATING_TYPES(input_coords.type(), "CoordsTranslateGPU_forward", ([&]{
        gpu_CoordsTranslateForward<scalar_t>(   input_coords.data<scalar_t>(),
                                                output_coords.data<scalar_t>(),
                                                T.data<scalar_t>(),
                                                num_atoms.data<int>(),
                                                batch_size,
                                                atoms_stride );
    }));
    
}

void CoordsTranslateGPU_backward(   torch::Tensor grad_output_coords, 
                                    torch::Tensor grad_input_coords,
                                    torch::Tensor T,
                                    torch::Tensor num_atoms
                                ){
    CHECK_GPU_INPUT(grad_output_coords);
    CHECK_GPU_INPUT(grad_input_coords);
    CHECK_GPU_INPUT(T);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(grad_output_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    int batch_size = grad_output_coords.size(0);
    int atoms_stride = grad_output_coords.size(1)/3;
    AT_DISPATCH_FLOATING_TYPES(grad_output_coords.type(), "CoordsTranslateGPU_backward", ([&]{
        gpu_CoordsTranslateBackward<scalar_t>(  grad_output_coords.data<scalar_t>(),
                                                grad_input_coords.data<scalar_t>(),
                                                T.data<scalar_t>(),
                                                num_atoms.data<int>(),
                                                batch_size,
                                                atoms_stride );
    }));
    
}

void CoordsRotateGPU_forward(   torch::Tensor input_coords, 
                                torch::Tensor output_coords,
                                torch::Tensor R,
                                torch::Tensor num_atoms
                            ){
    CHECK_GPU_INPUT(input_coords);
    CHECK_GPU_INPUT(output_coords);
    CHECK_GPU_INPUT(R);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(input_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    
    int batch_size = input_coords.size(0);
    int atoms_stride = input_coords.size(1)/3;
    AT_DISPATCH_FLOATING_TYPES(input_coords.type(), "CoordsRotateGPU_forward", ([&]{
        gpu_CoordsRotateForward<scalar_t>(      input_coords.data<scalar_t>(),
                                                output_coords.data<scalar_t>(),
                                                R.data<scalar_t>(),
                                                num_atoms.data<int>(),
                                                batch_size,
                                                atoms_stride );
    }));
}
void CoordsRotateGPU_backward(  torch::Tensor grad_output_coords, 
                                torch::Tensor grad_input_coords,
                                torch::Tensor R,
                                torch::Tensor num_atoms){
    CHECK_GPU_INPUT(grad_output_coords);
    CHECK_GPU_INPUT(grad_input_coords);
    CHECK_GPU_INPUT(R);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(grad_output_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    int batch_size = grad_output_coords.size(0);
    int atoms_stride = grad_output_coords.size(1)/3;
    AT_DISPATCH_FLOATING_TYPES(grad_output_coords.type(), "CoordsRotateGPU_backward", ([&]{
        gpu_CoordsRotateBackward<scalar_t>(     grad_output_coords.data<scalar_t>(),
                                                grad_input_coords.data<scalar_t>(),
                                                R.data<scalar_t>(),
                                                num_atoms.data<int>(),
                                                batch_size,
                                                atoms_stride );
    }));
    
}

void Coords2CenterGPU_forward(  torch::Tensor input_coords, 
                                torch::Tensor output_T,
                                torch::Tensor num_atoms
                            ){
    CHECK_GPU_INPUT(input_coords);
    CHECK_GPU_INPUT(output_T);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(input_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    
    int batch_size = input_coords.size(0);
    int atoms_stride = input_coords.size(1)/3;
    AT_DISPATCH_FLOATING_TYPES(input_coords.type(), "Coords2CenterGPU_forward", ([&]{
        gpu_Coords2CenterForward<scalar_t>(     input_coords.data<scalar_t>(),
                                                output_T.data<scalar_t>(),
                                                num_atoms.data<int>(),
                                                batch_size,
                                                atoms_stride );
    }));
}
void Coords2CenterGPU_backward( torch::Tensor grad_output_T, 
                                torch::Tensor grad_input_coords,
                                torch::Tensor num_atoms){
    CHECK_GPU_INPUT(grad_output_T);
    CHECK_GPU_INPUT(grad_input_coords);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(grad_input_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    int batch_size = grad_input_coords.size(0);
    int atoms_stride = grad_input_coords.size(1)/3;
    AT_DISPATCH_FLOATING_TYPES(grad_output_T.type(), "Coords2CenterGPU_backward", ([&]{
        gpu_Coords2CenterBackward<scalar_t>(    grad_output_T.data<scalar_t>(),
                                                grad_input_coords.data<scalar_t>(),
                                                num_atoms.data<int>(),
                                                batch_size,
                                                atoms_stride );
    }));
}

