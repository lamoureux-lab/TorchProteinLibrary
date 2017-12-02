#include <iostream>
#include <fstream>
#include <exception>
#include <limits>
#include <math.h>
#include <set>

#include "cAngles2BMatrix.h"
#include "../cABModelCUDAKernels.h"

cAngles2BMatrix::cAngles2BMatrix( THCState *state, int angles_length){
    this->angles_length = angles_length;

    this->d_alpha = THCudaTensor_new(state);
    this->d_beta = THCudaTensor_new(state);
    this->d_B_rot = THCudaTensor_new(state);
	this->d_B_bend = THCudaTensor_new(state);
    
    
}
cAngles2BMatrix::~cAngles2BMatrix(){
    THCudaTensor_free(state, this->d_alpha);
    THCudaTensor_free(state, this->d_beta);
    THCudaTensor_free(state, this->d_B_rot);
	THCudaTensor_free(state, this->d_B_bend);
    
}
 void cAngles2BMatrix::computeForward( THCudaTensor *input_angles,     // input angles, 2 x maxlen float tensor
                                        THCudaTensor *input_coords,       // output coords, 3 x maxlen + 1
                                        THCudaTensor *output_B){

    THCudaTensor_select(state, this->d_alpha, input_angles, 0, 0);
    THCudaTensor_select(state, this->d_beta, input_angles, 0, 1);
    THCudaTensor_select(state, this->d_B_bend, output_B, 0, 0);
    THCudaTensor_select(state, this->d_B_rot, output_B, 0, 1);
    
    cpu_computeBMatrix(	THCudaTensor_data(state, this->d_alpha), THCudaTensor_data(state, this->d_beta),
                        THCudaTensor_data(state, input_coords),
                        THCudaTensor_data(state, this->d_B_bend),
                        THCudaTensor_data(state, this->d_B_rot),
                        this->angles_length);

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
	// cpu_computeCoordinates(	THCudaTensor_data(state, this->d_alpha), 
	// 						THCudaTensor_data(state, this->d_beta), 
	// 						THCudaTensor_data(state, output_coords), 
	// 						THCudaTensor_data(state, this->d_A), 
	// 						this->angles_length);
    // std::cout<<"computed"<<std::endl;
    

}   

// void cAngles2CoordsAB::computeBackward(THCudaTensor *gradInput,            //output gradient of the input angles
//                                        THCudaTensor *gradOutput_coords ){  //input gradient of the coordinates
                                    
//     // THCudaTensor_zero(state, this->d_drdalpha);
//     // THCudaTensor_zero(state, this->d_drdbeta);
//     // for(int i=0;i<gradOutput_coords->size[0];i++){
//     //     std::cout<<i<<" "<<THCudaTensor_get1d(state, gradOutput_coords, i)<<std::endl;
//     // }
//      // Computing the derivative vectors of coordinates wrt alpha and beta
//      cpu_computeDerivatives(	THCudaTensor_data(state, this->d_alpha), 
// 							THCudaTensor_data(state, this->d_beta), 
// 							THCudaTensor_data(state, this->d_drdalpha), 
// 							THCudaTensor_data(state, this->d_drdbeta), 
// 							THCudaTensor_data(state, this->d_A), 
// 							this->angles_length);
//     // for(int i=0;i<gradOutput_coords->size[0];i++){
//     //     std::cout<<i<<" "<<THCudaTensor_get1d(state, gradOutput_coords, i)<<std::endl;
//     // }
//     // Backward pass from gradients of coordinates to the gradients of angles
// 	THCudaTensor_select(state, d_dalpha, gradInput, 0, 0);
// 	THCudaTensor_select(state, d_dbeta, gradInput, 0, 1);
// 	cpu_backwardFromCoords( THCudaTensor_data(state, this->d_dalpha), 
// 							THCudaTensor_data(state, this->d_dbeta), 
// 							THCudaTensor_data(state, gradOutput_coords), //1d Tensor: 3 x number of atoms 
// 							THCudaTensor_data(state, this->d_drdalpha), 
// 							THCudaTensor_data(state, this->d_drdbeta), 
// 							this->angles_length);
// }