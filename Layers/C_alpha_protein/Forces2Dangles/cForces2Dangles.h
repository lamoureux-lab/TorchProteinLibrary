#ifndef CFORCES2DANGLES_H_
#define CFORCES2DANGLES_H_
#include <TH.h>
#include <THC.h>


class cForces2Dangles{

	THCudaTensor *d_alpha, *d_beta, *d_B_rot, *d_B_bend;	
	THCState *state;

	int angles_length;
	    	
public:

	cForces2Dangles(    THCState *state, 
                        int angles_length          // input actual number of angles in the sequence
                    );
        
    void computeB(  THCudaTensor *input_angles,     // input angles, 2 x maxlen float tensor
                    THCudaTensor *input_coords,      // output coords, 3 x maxlen + 1
                    THCudaTensor *output_B      // output B-matrix 2 x 3 x num_atoms x num_angles
                    );

    void computeForward(THCudaTensor *input_forces, THCudaTensor *output_dangles);
                        
	// void computeBackward(   THCudaTensor *gradInput,            //output gradient of the input angles
    //                         THCudaTensor *gradOutput_coords    //input gradient of the coordinates
    //                         );
                            
	~cForces2Dangles();

};

#endif