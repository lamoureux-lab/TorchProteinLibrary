#ifndef CDDIST2FORCES_H_
#define CDDIST2FORCES_H_
#include <TH.h>
#include <THC.h>

class cDdist2Forces{
	THCState *state;
    THCudaTensor *coords;
	int angles_length;

public:

	cDdist2Forces(  THCState *state,
                    THCudaTensor *coords,
                    int angles_length          // input actual number of angles in the sequence
                    );

    void computeForward(THCudaTensor *input_ddist, THCudaTensor *output_forces);
                        
	// void computeBackward(   THCudaTensor *gradInput,            //output gradient of the input angles
    //                         THCudaTensor *gradOutput_coords    //input gradient of the coordinates
    //                         );
                            
	~cDdist2Forces();

};

#endif