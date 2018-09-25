#include <TH/TH.h>
#include <THC/THC.h>
#include "cPDBLoader.h"
#include <iostream>
#include <string>
#include <Kernels.h>

extern THCState *state;

extern "C" {
        void selectVolume_forward(  THCudaTensor *volume,
                                    THCudaTensor *coords,
                                    THCudaIntTensor *num_atoms,
                                    THCudaTensor *features,
                                    float res
                                ){
        if(coords->nDimension == 2){
            int batch_size = coords->size[0];
            int num_features = features->size[1];
            int spatial_dim = volume->size[2];
            int max_num_atoms = coords->size[1]/3;

            #pragma omp parallel for //num_threads(10)
            for(int i=0; i<batch_size; i++){
                THCudaTensor *single_volume = THCudaTensor_new(state);
                THCudaTensor *single_coords = THCudaTensor_new(state);
                THCudaTensor *single_features = THCudaTensor_new(state);
                
                THCudaTensor_select(state, single_volume, volume, 0, i);
                THCudaTensor_select(state, single_coords, coords, 0, i);
                THCudaTensor_select(state, single_features, features, 0, i);
                
                int single_num_atoms = THCudaIntTensor_get1d(state, num_atoms, i);
                gpu_selectFromTensor(THCudaTensor_data(state, single_features), num_features,
                                     THCudaTensor_data(state, single_volume), spatial_dim,
                                     THCudaTensor_data(state, single_coords), single_num_atoms, max_num_atoms, res);
                                     
                THCudaTensor_free(state, single_volume);
                THCudaTensor_free(state, single_coords);
                THCudaTensor_free(state, single_features);        
            }
        }else{
            std::cout<<"Not implemented\n";
        }
    }
}