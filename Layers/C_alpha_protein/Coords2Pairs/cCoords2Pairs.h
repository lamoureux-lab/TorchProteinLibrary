#ifndef CCOORDS2PAIRS_H_
#define CCOORDS2PAIRS_H_
#include <TH.h>
#include <THC.h>


class cCoords2Pairs{

	THCState *state;

    THCudaTensor *d_dcoordinates, *d_coordinates; //coordinates and their derivatives
    THCudaTensor *d_paircoords, *d_dpaircoords; //pairwise coordinates and their derivatives

	int angles_length, stride;
    	
public:

	cCoords2Pairs(  THCState *state, 
                    int angles_length,              // input actual number of angles in the sequence
                    int stride                      // stride in the output array (default angles_length)
                   );
	
    void computeForward(    THCudaTensor *input_coords,     // input coords, 3 x maxlen + 1 
                            THCudaTensor *output_pairs      // output pair coords, 3 x maxlen + 1 x maxlen + 1
                            );
                        
	void computeBackward(   THCudaTensor *gradInput_coords,   //output gradient of the input coords
                            THCudaTensor *gradOutput_pairs    //input gradient of the pairwise coordinates
                            );
                            
	~cCoords2Pairs();

};

#endif