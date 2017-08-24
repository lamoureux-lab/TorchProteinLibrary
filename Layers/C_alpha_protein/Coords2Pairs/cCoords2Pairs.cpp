#include <iostream>
#include <fstream>
#include <exception>
#include <limits>
#include <math.h>
#include <set>

#include "cCoords2Pairs.h"
#include "../cTensorProteinCUDAKernels.h"
#include "../cPairwisePotentialsKernels.h"

cCoords2Pairs::cCoords2Pairs( THCState *state, int angles_length, int stride){
    this->angles_length = angles_length;
    this->stride = stride;
}
cCoords2Pairs::~cCoords2Pairs(){
    
}
 void cCoords2Pairs::computeForward(    THCudaTensor *input_coords,     // input coords, 3 x maxlen + 1 
                                        THCudaTensor *output_pairs      // output pair coords, 3 x maxlen + 1 x maxlen + 1
                                        ){
    
	// Forward coordinates -> pairwise coordinates
    cpu_computePairCoordinates( THCudaTensor_data(state, input_coords),
                                THCudaTensor_data(state, output_pairs),
                                this->angles_length,
								this->stride );
}   

void cCoords2Pairs::computeBackward(    THCudaTensor *gradInput_coords,   //output gradient of the input coords
                                        THCudaTensor *gradOutput_pairs    //input gradient of the pairwise coordinates
                                        ){  
   // Backward from pairwise coordinates to coordinates
    cpu_backwardFromPairCoordinates(    THCudaTensor_data(state, gradInput_coords), 
                                		THCudaTensor_data(state, gradOutput_pairs),
                               		 	this->angles_length,
										this->stride);
}