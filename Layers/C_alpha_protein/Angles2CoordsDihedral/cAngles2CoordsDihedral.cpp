#include <iostream>
#include <fstream>
#include <exception>
#include <limits>
#include <math.h>
#include <set>

#include "cAngles2CoordsDihedral.h"
#include "../cTensorProteinCUDAKernels.h"
#include "../cPairwisePotentialsKernels.h"

cAngles2CoordsDihedral::cAngles2CoordsDihedral( THCState *state, THCudaTensor *A, THCudaTensor *input_angles, THCudaTensor *dr_dangle, 
                                float R, int angles_length){
    d_A = A;
    this->R = R;
    this->angles_length = angles_length;

    this->d_alpha = THCudaTensor_new(state);
	this->d_beta = THCudaTensor_new(state);
    this->d_dalpha = THCudaTensor_new(state);
	this->d_dbeta = THCudaTensor_new(state);
    THCudaTensor_select(state, this->d_alpha, input_angles, 0, 0);
	THCudaTensor_select(state, this->d_beta, input_angles, 0, 1);

    if(dr_dangle!=NULL){
        //coordinates derivatives wrt angles
        this->d_drdalpha = THCudaTensor_new(state);
        this->d_drdbeta = THCudaTensor_new(state);
        THCudaTensor_select(state, this->d_drdalpha, dr_dangle, 0, 0);
        THCudaTensor_select(state, this->d_drdbeta, dr_dangle, 0, 1);
    }else{
        this->d_drdalpha = NULL;
        this->d_drdbeta = NULL;
    }
}
cAngles2CoordsDihedral::~cAngles2CoordsDihedral(){
    THCudaTensor_free(state, this->d_alpha);
	THCudaTensor_free(state, this->d_beta);
    THCudaTensor_free(state, this->d_dalpha);
	THCudaTensor_free(state, this->d_dbeta);
    if(this->d_drdalpha!=NULL)
        THCudaTensor_free(state, this->d_drdalpha);
    if(this->d_drdbeta!=NULL)
	    THCudaTensor_free(state, this->d_drdbeta);
}
 void cAngles2CoordsDihedral::computeForward(   THCudaTensor *input_angles,     // input angles, 2 x maxlen float tensor
                                        THCudaTensor *output_coords){      // output coords, 3 x maxlen + 1
    // std::cout<<"an_len="<<this->angles_length<<std::endl;
    // std::cout<<"a_len="<<this->d_A->size[0]<<std::endl;
    // std::cout<<"c_len="<<output_coords->size[0]<<std::endl;
    // std::cout<<"al_len="<<this->d_alpha->size[0]<<std::endl;
    // std::cout<<"be_len="<<this->d_beta->size[0]<<std::endl;
    // for(int i=0;i<output_coords->size[0];i++){
    //     std::cout<<i<<" "<<THCudaTensor_get1d(state, output_coords, i)<<std::endl;
    // }
    // THCudaTensor_zero(state, output_coords);
    // std::cout<<"zeroed"<<std::endl;
    // Forward pass angles -> coordinates
	cpu_computeCoordinatesDihedral(	THCudaTensor_data(state, this->d_alpha), 
							THCudaTensor_data(state, this->d_beta), 
							THCudaTensor_data(state, output_coords), 
							THCudaTensor_data(state, this->d_A), 
							this->angles_length, this->R);
    // std::cout<<"computed"<<std::endl;
}   

void cAngles2CoordsDihedral::computeBackward(   THCudaTensor *gradInput,            //output gradient of the input angles
                                        THCudaTensor *gradOutput_coords ){  //input gradient of the coordinates
                                    
    // THCudaTensor_zero(state, this->d_drdalpha);
    // THCudaTensor_zero(state, this->d_drdbeta);
    // for(int i=0;i<gradOutput_coords->size[0];i++){
    //     std::cout<<i<<" "<<THCudaTensor_get1d(state, gradOutput_coords, i)<<std::endl;
    // }
     // Computing the derivative vectors of coordinates wrt alpha and beta
    cpu_computeDerivativesDihedral(	THCudaTensor_data(state, this->d_alpha), 
							THCudaTensor_data(state, this->d_beta), 
							THCudaTensor_data(state, this->d_drdalpha), 
							THCudaTensor_data(state, this->d_drdbeta), 
							THCudaTensor_data(state, this->d_A), 
							this->angles_length, 
							this->R);
    // for(int i=0;i<gradOutput_coords->size[0];i++){
    //     std::cout<<i<<" "<<THCudaTensor_get1d(state, gradOutput_coords, i)<<std::endl;
    // }
    // Backward pass from gradients of coordinates to the gradients of angles
	THCudaTensor_select(state, d_dalpha, gradInput, 0, 0);
	THCudaTensor_select(state, d_dbeta, gradInput, 0, 1);
	cpu_backwardFromCoords( THCudaTensor_data(state, this->d_dalpha), 
							THCudaTensor_data(state, this->d_dbeta), 
							THCudaTensor_data(state, gradOutput_coords), //1d Tensor: 3 x number of atoms 
							THCudaTensor_data(state, this->d_drdalpha), 
							THCudaTensor_data(state, this->d_drdbeta), 
							this->angles_length);
}