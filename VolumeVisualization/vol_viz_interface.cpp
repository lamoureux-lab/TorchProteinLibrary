#include <TH.h>
#include <THC.h>
#include <GlutFramework.h>
#include <cVector3.h>
#include <cMarchingCubes.h>
#include <iostream>

using namespace glutFramework;

extern THCState *state;
void toCPUTensor(THCState*state, THFloatTensor *cpu_T, THCudaTensor *gpu_T){
    uint size = 1;
    for(int i=0; i<gpu_T->nDimension; i++)
        size *= gpu_T->size[i];
    cudaMemcpy( THFloatTensor_data(cpu_T), 
                THCudaTensor_data(state, gpu_T),
                size*sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C" {
    
    int VisualizeVolume4d(THCudaTensor *gpu_volume){
        GlutFramework framework;
        int spatial_dim = gpu_volume->size[1];
        THFloatTensor *c_volume = THFloatTensor_newWithSize4d(gpu_volume->size[0], spatial_dim, spatial_dim, spatial_dim);

        toCPUTensor(state, c_volume, gpu_volume);   
        cVolume v(c_volume);
        Vector<double> lookAtPos(spatial_dim/2,spatial_dim/2,spatial_dim/2);
        framework.setLookAt(spatial_dim, spatial_dim, spatial_dim, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);
        framework.addObject(&v);
        framework.startFramework(0, NULL);     
        THFloatTensor_free(c_volume);
    }
}