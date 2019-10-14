
void gpu_computePartialSumFaces(	float *coords,
                                    float *assigned_params,
                                    int num_atoms, 
                                    float *volume,
                                    int box_size,
                                    float res,
                                    float stern_size);

void gpu_computeSumCells(	float *coords,
                            float *assigned_params,
                            int num_atoms, 
                            float *volume,
                            int box_size,
                            float res);

void gpu_computePhi(    float *Q, float *Eps, float *Phi, int box_size, 
                        float res, float kappa02);