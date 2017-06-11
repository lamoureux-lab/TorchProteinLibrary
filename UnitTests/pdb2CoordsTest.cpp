#include <TH.h>
#include <THC.h>
#include <GlutFramework.h>
#include <cProteinLoader.h>
#include <cVector3.h>
#include <cMarchingCubes.h>

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
    char* filename = "/home/lupoglaz/ProteinsDataset/CASP_SCWRL/T0653/FALCON-TOPO-X_TS5";
	pL.loadPDB(filename);
    pL.assignAtomTypes(2);
    
    THCudaTensor *coords = THCudaTensor_newWithSize1d(state, pL.r.size()*3);
    THCudaIntTensor *offsets = THCudaIntTensor_newWithSize1d(state, 11);
    THCudaIntTensor *num_coords = THCudaIntTensor_newWithSize1d(state, 11);
	

    THFloatTensor *c_coords = THFloatTensor_newWithSize1d(pL.r.size()*3);
    THIntTensor *c_offsets = THIntTensor_newWithSize1d(11);
    THIntTensor *c_num_coords = THIntTensor_newWithSize1d(11);
    
    pL.pdb2Coords(state, filename, coords, offsets, num_coords, spatial_dim, resolution, false, false);
	
    toCPUTensor(state, c_coords, coords);
    toCPUTensor(state, c_offsets, offsets);
    toCPUTensor(state, c_num_coords, num_coords);

    for(int i=0; i<11; i++){
        std::cout<<"Atom type = "<<i;
        int num_coords_i = THIntTensor_get1d(c_num_coords, i);
        int offset = THIntTensor_get1d(c_offsets, i);
        std::cout<<" num_coords = "<<num_coords_i<<" offset = "<<offset<<"\n";
        for(int j=0; j<num_coords_i; j+=3){
            float x = THFloatTensor_get1d(c_coords, j + offset);
            float y = THFloatTensor_get1d(c_coords, j + offset + 1);
            float z = THFloatTensor_get1d(c_coords, j + offset + 2);
            std::cout<<"Atom: "<<x<<", "<<y<<", "<<z<<"\n";
        }
    }
    
    THCudaShutdown(state);
	free(state);
	
	return 0;
}