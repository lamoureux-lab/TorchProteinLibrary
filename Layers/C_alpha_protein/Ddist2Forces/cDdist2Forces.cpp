#include <iostream>
#include <fstream>
#include <exception>
#include <limits>
#include <math.h>
#include <set>

#include "cDdist2Forces.h"
#include "../cABModelCUDAKernels.h"

cDdist2Forces::cDdist2Forces( THCState *state, THCudaTensor *coords, int angles_length){
    this->angles_length = angles_length;
    this->coords = coords;
    
}
cDdist2Forces::~cDdist2Forces(){
    
}

void cDdist2Forces::computeForward( THCudaTensor *input_ddist, 
                                    THCudaTensor *output_forces){

    cpu_computeForces(	THCudaTensor_data(state, input_ddist),
                        THCudaTensor_data(state, output_forces),
                        THCudaTensor_data(state, this->coords),
                        this->angles_length);

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