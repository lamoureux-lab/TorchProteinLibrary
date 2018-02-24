#include <TH/TH.h>
#include <THC/THC.h>
#include "cPDBLoader.h"
#include <iostream>
#include <string>
#include <Kernels.h>

extern THCState *state;

extern "C" {

    void TypedCoords2Volume_forward(    THCudaDoubleTensor *input_coords,
                                        THCudaTensor *volume,
                                        THCudaIntTensor *num_atoms_of_type,
                                        THCudaIntTensor *offsets){
        if(input_coords->nDimension == 1){
            uint num_atom_types=11;
                       
            gpu_computeCoords2Volume(   THCudaDoubleTensor_data(state, input_coords), 
                                        (uint*)THCudaIntTensor_data(state, num_atoms_of_type), 
                                        (uint*)THCudaIntTensor_data(state, offsets), 
                                        THCudaTensor_data(state, volume), volume->size[1], num_atom_types, 1.0);
            
        }else if(input_coords->nDimension == 2){
            throw std::string("Not implemented");
        }
        
    }
    void TypedCoords2Volume_backward(   THCudaTensor *grad_volume,
                                        THCudaDoubleTensor *grad_coords,
                                        THCudaDoubleTensor *coords,
                                        THCudaIntTensor *num_atoms_of_type,
                                        THCudaIntTensor *offsets){
        if(grad_coords->nDimension == 1){
            uint num_atom_types=11;
            
            gpu_computeVolume2Coords(   THCudaDoubleTensor_data(state, coords), 
                                        THCudaDoubleTensor_data(state, grad_coords),
                                        (uint*)THCudaIntTensor_data(state, num_atoms_of_type),
                                        (uint*)THCudaIntTensor_data(state, offsets), 
                                        THCudaTensor_data(state, grad_volume), 
                                        grad_volume->size[1], num_atom_types, 1.0);

            
        }else if(grad_coords->nDimension == 2){
            throw std::string("Not implemented");
        }
        
    }
}