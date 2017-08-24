#include <THC/THC.h>


void cpu_computePairwiseForcesPP(   float *d_paircoords,    //input: pairwise coordinates 3 x maxlen x maxlen
                                    int *input_types,       //input: atom types maxlen
                                    float *output_forces,   //output: forces maxlen x 3
                                    float *potentials,      //params: potentials num_types x num_types x num_bins
                                    int num_types,
                                    int num_bins,
                                    float resolution,
                                    int L,int Lmax);    

void cpu_computeGradientForcesPP(   float *forces_grads,    //input: gradients of forces 3 x maxlen
                                    float *d_dpaircoords,   //output: maxlen x 3
                                    float *d_paircoords,      //params: coordinates maxlen x 3
                                    float *potentials,      //params: potentials num_types x num_types x num_bins
                                    int *input_types,       //params: atom types maxlen
                                    int num_types,
                                    int num_bins,
                                    float resolution,
                                    int L, int Lmax);  

void cpu_computeGradientPotentials(     float *forces_grads,    //input: gradients of forces 3 x maxlen
                                        float *d_dpotentials,   //output: num_types x num_types x num_bins
                                        float *d_paircoords,      //params: coordinates maxlen x 3
                                        int *input_types,       //params: atom types maxlen
                                        int num_types,
                                        int num_bins,
                                        float resolution,
                                        int L, int Lmax);  


void cpu_computePairwiseDistributions(float *d_paircoords,    //input: coordinates 3 x maxlen x maxlen
                                    int *input_types,       //input: atom types maxlen
                                    float *output_dist,   //output: forces maxlen x 3
                                    int num_types,
                                    int num_bins,
                                    float resolution,
                                    int L, int Lmax);  

void cpu_backwardPairwiseDistributions( float *gradInput_pairs,    //input: coordinates 3 x maxlen x maxlen
                                        float *gradOutput_dist,       //input: atom types maxlen
                                        int *input_types,   //output: forces maxlen x 3
                                        float *input_pairs,
                                        int num_types,
                                        int num_bins,
                                        float resolution,
                                        int L, int Lmax);   