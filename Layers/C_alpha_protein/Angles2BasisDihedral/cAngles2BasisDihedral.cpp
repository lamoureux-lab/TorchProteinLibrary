#include <iostream>
#include <fstream>
#include <exception>
#include <limits>
#include <math.h>
#include <set>

#include "cAngles2BasisDihedral.h"
#include "../cTensorProteinCUDAKernels.h"
#include "../cPairwisePotentialsKernels.h"

cAngles2BasisDihedral::cAngles2BasisDihedral( THCState *state, THCudaTensor *B, THCudaTensor *input_angles, THCudaTensor *daxis_dangle, 
                                float R, int angles_length){
    this->d_B = B;
    this->R = R;
    this->angles_length = angles_length;

    this->d_alpha = THCudaTensor_new(state);
	this->d_beta = THCudaTensor_new(state);
    this->d_dalpha = THCudaTensor_new(state);
	this->d_dbeta = THCudaTensor_new(state);
    THCudaTensor_select(state, this->d_alpha, input_angles, 0, 0);
	THCudaTensor_select(state, this->d_beta, input_angles, 0, 1);

    this->d_daxisx = THCudaTensor_new(state);
    this->d_daxisy = THCudaTensor_new(state);
    this->d_daxisz = THCudaTensor_new(state);
    
    this->d_axisx = THCudaTensor_new(state);
    this->d_axisy = THCudaTensor_new(state);
    this->d_axisz = THCudaTensor_new(state);

    if(daxis_dangle!=NULL){
        //coordinates derivatives wrt angles
        this->dax_dalpha = THCudaTensor_new(state);
        this->dax_dbeta = THCudaTensor_new(state);
        this->day_dalpha = THCudaTensor_new(state);
        this->day_dbeta = THCudaTensor_new(state);
        this->daz_dalpha = THCudaTensor_new(state);
        this->daz_dbeta = THCudaTensor_new(state);
        THCudaTensor_select(state, this->dax_dalpha, daxis_dangle, 0, 0);
        THCudaTensor_select(state, this->dax_dbeta, daxis_dangle, 0, 1);
        THCudaTensor_select(state, this->day_dalpha, daxis_dangle, 0, 2);
        THCudaTensor_select(state, this->day_dbeta, daxis_dangle, 0, 3);
        THCudaTensor_select(state, this->daz_dalpha, daxis_dangle, 0, 4);
        THCudaTensor_select(state, this->daz_dbeta, daxis_dangle, 0, 5);
    }else{
        this->dax_dalpha = NULL;
        this->dax_dbeta = NULL;
        this->day_dalpha = NULL;
        this->day_dbeta = NULL;
        this->daz_dalpha = NULL;
        this->daz_dbeta = NULL;
    }
}
cAngles2BasisDihedral::~cAngles2BasisDihedral(){
    THCudaTensor_free(state, this->d_alpha);
	THCudaTensor_free(state, this->d_beta);
    THCudaTensor_free(state, this->d_dalpha);
	THCudaTensor_free(state, this->d_dbeta);
    if(this->dax_dalpha!=NULL)
        THCudaTensor_free(state, this->dax_dalpha);
    if(this->dax_dbeta!=NULL)
	    THCudaTensor_free(state, this->dax_dbeta);
    if(this->day_dalpha!=NULL)
        THCudaTensor_free(state, this->day_dalpha);
    if(this->day_dbeta!=NULL)
	    THCudaTensor_free(state, this->day_dbeta);
    if(this->daz_dalpha!=NULL)
        THCudaTensor_free(state, this->daz_dalpha);
    if(this->daz_dbeta!=NULL)
	    THCudaTensor_free(state, this->daz_dbeta);
    
    THCudaTensor_free(state, this->d_daxisx);
	THCudaTensor_free(state, this->d_daxisy);
    THCudaTensor_free(state, this->d_daxisz);
    THCudaTensor_free(state, this->d_axisx);
	THCudaTensor_free(state, this->d_axisy);
    THCudaTensor_free(state, this->d_axisz);
}
 void cAngles2BasisDihedral::computeForward(   THCudaTensor *input_angles,     // input angles, 2 x maxlen float tensor
                                        THCudaTensor *output_basis){      // output coords, 3 x maxlen + 1
    // Forward pass angles -> coordinates
	THCudaTensor_select(state, this->d_axisx, output_basis, 0, 0);
	THCudaTensor_select(state, this->d_axisy, output_basis, 0, 1);
	THCudaTensor_select(state, this->d_axisz, output_basis, 0, 2);
    // Forward pass angles -> basis
	cpu_computeBasisDihedral(		THCudaTensor_data(state, this->d_alpha), 
							THCudaTensor_data(state, this->d_beta), 
							THCudaTensor_data(state, this->d_axisx), 
							THCudaTensor_data(state, this->d_axisy), 
							THCudaTensor_data(state, this->d_axisz),
							THCudaTensor_data(state, this->d_B),
							this->angles_length);
}   

void cAngles2BasisDihedral::computeBackward(   THCudaTensor *gradInput_angles,            //output gradient of the input angles
                                        THCudaTensor *gradOutput_basis ){  //input gradient of the coordinates
   
   // Computing derivative of basis vectors wrt alpha and beta
    THCudaTensor_select(state, this->d_dalpha, gradInput_angles, 0, 0);
	THCudaTensor_select(state, this->d_dbeta, gradInput_angles, 0, 1);
    THCudaTensor_select(state, this->d_daxisx, gradOutput_basis, 0, 0);
	THCudaTensor_select(state, this->d_daxisy, gradOutput_basis, 0, 1);
	THCudaTensor_select(state, this->d_daxisz, gradOutput_basis, 0, 2);
	cpu_computeBasisGradientsDihedral(	THCudaTensor_data(state, this->d_alpha),
								THCudaTensor_data(state, this->d_beta),
								THCudaTensor_data(state, this->dax_dalpha), 
								THCudaTensor_data(state, this->dax_dbeta), 
								THCudaTensor_data(state, this->day_dalpha), 
								THCudaTensor_data(state, this->day_dbeta), 
								THCudaTensor_data(state, this->daz_dalpha), 
								THCudaTensor_data(state, this->daz_dbeta), 
								THCudaTensor_data(state, this->d_B),
								this->angles_length);

    // Backward pass from basis to angles derivatives
	cpu_backwardFromBasis(		THCudaTensor_data(state, this->d_dalpha),
								THCudaTensor_data(state, this->d_dbeta),
								THCudaTensor_data(state, this->d_daxisx),
								THCudaTensor_data(state, this->d_daxisy),
								THCudaTensor_data(state, this->d_daxisz),
								THCudaTensor_data(state, this->dax_dalpha), 
								THCudaTensor_data(state, this->dax_dbeta), 
								THCudaTensor_data(state, this->day_dalpha), 
								THCudaTensor_data(state, this->day_dbeta), 
								THCudaTensor_data(state, this->daz_dalpha), 
								THCudaTensor_data(state, this->daz_dbeta),
								this->angles_length);	
}