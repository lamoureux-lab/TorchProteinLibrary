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
        int num_atom_types=11;
        if(input_coords->nDimension == 1){
            
                       
            gpu_computeCoords2Volume(   THCudaDoubleTensor_data(state, input_coords), 
                                        THCudaIntTensor_data(state, num_atoms_of_type), 
                                        THCudaIntTensor_data(state, offsets), 
                                        THCudaTensor_data(state, volume), volume->size[1], num_atom_types, 1.0);
            
        }else if(input_coords->nDimension == 2){
            int batch_size = input_coords->size[0];

            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THCudaIntTensor *single_num_atoms_of_type = THCudaIntTensor_new(state);
                THCudaIntTensor *single_offsets = THCudaIntTensor_new(state);
                THCudaIntTensor_select(state, single_num_atoms_of_type, num_atoms_of_type, 0, i);
                THCudaIntTensor_select(state, single_offsets, offsets, 0, i);
                THCudaTensor *single_volume = THCudaTensor_new(state);
                THCudaDoubleTensor *single_input_coords = THCudaDoubleTensor_new(state);
                THCudaTensor_select(state, single_volume, volume, 0, i);
                THCudaDoubleTensor_select(state, single_input_coords, input_coords, 0, i);

                gpu_computeCoords2Volume(   THCudaDoubleTensor_data(state, single_input_coords), 
                                            THCudaIntTensor_data(state, single_num_atoms_of_type), 
                                            THCudaIntTensor_data(state, single_offsets), 
                                            THCudaTensor_data(state, single_volume), single_volume->size[1], num_atom_types, 1.0);
                
                THCudaTensor_free(state, single_volume);
                THCudaDoubleTensor_free(state, single_input_coords);
                THCudaIntTensor_free(state, single_num_atoms_of_type);
                THCudaIntTensor_free(state, single_offsets);
            }
            
        }else{
            throw std::string("Not implemented");
        }
        
    }
    void TypedCoords2Volume_backward(   THCudaTensor *grad_volume,
                                        THCudaDoubleTensor *grad_coords,
                                        THCudaDoubleTensor *coords,
                                        THCudaIntTensor *num_atoms_of_type,
                                        THCudaIntTensor *offsets){
        int num_atom_types=11;
        if(grad_coords->nDimension == 1){
                        
            gpu_computeVolume2Coords(   THCudaDoubleTensor_data(state, coords), 
                                        THCudaDoubleTensor_data(state, grad_coords),
                                        THCudaIntTensor_data(state, num_atoms_of_type),
                                        THCudaIntTensor_data(state, offsets), 
                                        THCudaTensor_data(state, grad_volume), 
                                        grad_volume->size[1], num_atom_types, 1.0);

            
        }else if(grad_coords->nDimension == 2){
            int batch_size = grad_coords->size[0];

            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THCudaIntTensor *single_num_atoms_of_type = THCudaIntTensor_new(state);
                THCudaIntTensor *single_offsets = THCudaIntTensor_new(state);
                THCudaIntTensor_select(state, single_num_atoms_of_type, num_atoms_of_type, 0, i);
                THCudaIntTensor_select(state, single_offsets, offsets, 0, i);
                THCudaTensor *single_grad_volume = THCudaTensor_new(state);
                THCudaDoubleTensor *single_coords = THCudaDoubleTensor_new(state);
                THCudaDoubleTensor *single_grad_coords = THCudaDoubleTensor_new(state);
                THCudaTensor_select(state, single_grad_volume, grad_volume, 0, i);
                THCudaDoubleTensor_select(state, single_grad_coords, grad_coords, 0, i);
                THCudaDoubleTensor_select(state, single_coords, coords, 0, i);

                gpu_computeVolume2Coords(   THCudaDoubleTensor_data(state, single_coords), 
                                            THCudaDoubleTensor_data(state, single_grad_coords),
                                            THCudaIntTensor_data(state, single_num_atoms_of_type),
                                            THCudaIntTensor_data(state, single_offsets), 
                                            THCudaTensor_data(state, single_grad_volume), 
                                            single_grad_volume->size[1], num_atom_types, 1.0);
                
                THCudaTensor_free(state, single_grad_volume);
                THCudaDoubleTensor_free(state, single_coords);
                THCudaDoubleTensor_free(state, single_grad_coords);
                THCudaIntTensor_free(state, single_num_atoms_of_type);
                THCudaIntTensor_free(state, single_offsets);
            }
        }else{
            throw std::string("Not implemented");
        
        }
        
    }
}