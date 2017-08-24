#ifndef CANGLES2BASIS_H_
#define CANGLES2BASIS_H_
#include <TH.h>
#include <THC.h>


class cAngles2BasisDihedral{

	THCudaTensor *d_alpha, *d_beta, *d_B, *d_dalpha, *d_dbeta; // angles derivatives
	THCudaTensor *d_axisx, *d_axisy, *d_axisz;
    THCudaTensor *d_daxisx, *d_daxisy, *d_daxisz;
    THCudaTensor *dax_dalpha, *day_dalpha, *daz_dalpha;
    THCudaTensor *dax_dbeta, *day_dbeta, *daz_dbeta;
	THCState *state;

	int angles_length;
	float R;
    	
public:

	cAngles2BasisDihedral(  THCState *state, 
                    THCudaTensor *B,            //9 x max_angles
                    THCudaTensor *input_angles, //future input
                    THCudaTensor *daxis_dangle,    //6 x 3 x max_atoms x max_angles
                    float R,                        //bond length
                    int angles_length              // input actual number of angles in the sequence
                   );
	
    void computeForward(    THCudaTensor *input_angles,     // input angles, 2 x maxlen float tensor
                            THCudaTensor *output_basis      // output coords, 3 x maxlen + 1
                            );
                        
	void computeBackward(   THCudaTensor *gradInput_angles,            //output gradient of the input angles
                            THCudaTensor *gradOutput_basis    //input gradient of the coordinates
                            );
                            
	~cAngles2BasisDihedral();

};

#endif