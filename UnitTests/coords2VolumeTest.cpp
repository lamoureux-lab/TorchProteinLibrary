#include <TH.h>
#include <THC.h>
#include <GlutFramework.h>
#include <cProteinLoader.h>
#include <cVector3.h>
#include <cMarchingCubes.h>
#include <cCoords2Volume.h>

using namespace glutFramework;

void toGPUTensor(THCState*state, float *cpu_T, THCudaTensor *gpu_T){
    uint size = 1;
    for(int i=0; i<gpu_T->nDimension; i++)
        size *= gpu_T->size[i];
    cudaMemcpy( THCudaTensor_data(state, gpu_T), 
                cpu_T,
                size*sizeof(float), cudaMemcpyHostToDevice);
}

void toCPUTensor(THCState*state, THFloatTensor *cpu_T, THCudaTensor *gpu_T){
    uint size = 1;
    for(int i=0; i<gpu_T->nDimension; i++)
        size *= gpu_T->size[i];
    cudaMemcpy( THFloatTensor_data(cpu_T), 
                THCudaTensor_data(state, gpu_T),
                size*sizeof(float), cudaMemcpyDeviceToHost);
}

void toCPUTensor(THCState*state, THIntTensor *cpu_T, THCudaIntTensor *gpu_T){
    uint size = 1;
    for(int i=0; i<gpu_T->nDimension; i++)
        size *= gpu_T->size[i];
    cudaMemcpy( THIntTensor_data(cpu_T), 
                THCudaIntTensor_data(state, gpu_T),
                size*sizeof(int), cudaMemcpyDeviceToHost);
}

int main(int argc, char** argv)
{

    GlutFramework framework;
    THCState* state = (THCState*) malloc(sizeof(THCState));
	memset(state, 0, sizeof(THCState));
	THCudaInit(state);
    
    int spatial_dim = 120;
    float resolution = 1.0;
	cProteinLoader pL;
    char* filename = "/media/lupoglaz/a56f0954-3abe-49ae-a024-5c17afc19995/ProteinsDataset/3DRobot_set/1BYIA/decoy0_1.pdb";
	pL.loadPDB(filename);
    pL.assignAtomTypes(2);
    cCoords2Volume c2v(11, spatial_dim, resolution);
    
    THCudaTensor *coords = THCudaTensor_newWithSize1d(state, pL.r.size()*3);
    THCudaIntTensor *offsets = THCudaIntTensor_newWithSize1d(state, 11);
    THCudaIntTensor *num_coords = THCudaIntTensor_newWithSize1d(state, 11);
	THCudaTensor *volume = THCudaTensor_newWithSize4d(state, 11, spatial_dim, spatial_dim, spatial_dim);

    THFloatTensor *c_volume = THFloatTensor_newWithSize4d(11, spatial_dim, spatial_dim, spatial_dim);
    
    pL.pdb2Coords(state, filename, coords, offsets, num_coords, spatial_dim, resolution, false, false);
    c2v.computeForward(state, coords, offsets, num_coords, volume);
	
    toCPUTensor(state, c_volume, volume);

    cVolume v(c_volume);
    Vector<double> lookAtPos(60,60,60);
    framework.setLookAt(120, 120.0, 120.0, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);
	framework.addObject(&v);
    framework.startFramework(argc, argv);

    THCudaShutdown(state);
	free(state);
	
	return 0;
}