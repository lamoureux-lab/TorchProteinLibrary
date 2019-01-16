#include "volumeRotation_interface.h"
#include <RotateGrid.h>
#include <iostream>


void VolumeGenGrid( at::Tensor rotations, at::Tensor grid){
    // if( volume1.dtype() != at::kFloat || volume2.dtype() != at::kFloat || output.dtype() != at::kFloat){
    //     throw("Incorrect tensor types");
    // }
    if( (!rotations.type().is_cuda()) || (!grid.type().is_cuda()) ){
        throw("Incorrect device");
    }
    if(rotations.ndimension()!=3 || grid.ndimension()!=5){
        throw("incorrect input dimension");
    }
    int batch_size = rotations.size(0);
    int size = grid.size(1);
    cpu_RotateGrid(rotations.data<float>(), grid.data<float>(), batch_size, size);
}

