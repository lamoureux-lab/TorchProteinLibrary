#ifndef CPAIRS2DIST_H_
#define CPAIRS2DIST_H_
#include <TH.h>
#include <THC.h>


class cPairs2Dist{

	THCState *state;

    THCudaIntTensor *input_types;
    
	int angles_length, stride;
    int num_types, num_bins;
    float resolution;
    	
public:

	cPairs2Dist(    THCState *state, 
                    THCudaIntTensor *input_types,
                    int num_types,
                    int num_bins,
                    float resolution,
                    int angles_length,              // input actual number of angles in the sequence
                    int stride                      // stride in the output array (default angles_length)
                   );
	
    void computeForward(    THCudaTensor *input_pairs,     // input pair coords, 3 x maxlen + 1 x maxlen + 1
                            THCudaTensor *output_dist      // output 
                            );
                        
	void computeBackward(   THCudaTensor *gradInput_pairs,   //output gradient of the pairwise coordinates distributions
                            THCudaTensor *gradOutput_dist,    //input gradient 
                            THCudaTensor *input_pairs
                            );
                            
	~cPairs2Dist();

};

#endif