#ifndef CANGLES2COORDS_H_
#define CANGLES2COORDS_H_
#include <TH.h>
#include <THC.h>

#include <vector>
#include <GlutFramework.h>
using namespace glutFramework;

class cAngles2Backbone: public Object{

	THCudaTensor *d_phi, *d_psi, *d_A, *d_dphi, *d_dpsi; // angles derivatives
	THCudaTensor *d_drdphi, *d_drdpsi;	// coordinates derivatives
    THCudaTensor *d_r_calpha, *d_r_c, *d_r_n;	// coordinates derivatives

    THCudaTensor *angles;
    THCudaTensor *grad_angles, *grad_coords;
    
    THFloatTensor *c_phi, *c_psi, *c_A, *c_dphi, *c_dpsi; // angles derivatives
    THFloatTensor *c_drdphi, *c_drdpsi;	// coordinates derivatives
    THFloatTensor *c_r_calpha, *c_r_c, *c_r_n;	// coordinates derivatives



	THCState *state;

	int angles_length, num_atoms_calpha, num_atoms_c, num_atoms_n;
	float R;
    	
public:

	cAngles2Backbone(   THCState *state, 
                        THFloatTensor *input_angles    //future input
                    );
	
    void computeForward();
    void computeForwardCalpha();
                        
	void computeBackwardCalpha(   THCudaTensor *gradInput,            //output gradient of the input angles
                            THCudaTensor *gradOutput_coords    //input gradient of the coordinates
                            );

    void display();
                            
	~cAngles2Backbone();

    void toGPUTensor(THCState*state, THFloatTensor *cpu_T, THCudaTensor *gpu_T){
        uint size = 1;
        for(int i=0; i<gpu_T->nDimension; i++)
            size *= gpu_T->size[i];
        cudaMemcpy( THCudaTensor_data(state, gpu_T), 
                    THFloatTensor_data(cpu_T),
                    size*sizeof(float), cudaMemcpyHostToDevice);
    };

    void toCPUTensor(THCState*state, THCudaTensor *gpu_T, THFloatTensor *cpu_T){
        uint size = 1;
        for(int i=0; i<gpu_T->nDimension; i++)
            size *= gpu_T->size[i];
        cudaMemcpy( THFloatTensor_data(cpu_T), 
                    THCudaTensor_data(state, gpu_T),
                    size*sizeof(float), cudaMemcpyDeviceToHost);
    };


    void display_backbone();
    void display_Calpha();
};

#endif