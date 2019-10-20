
void gpu_computePartialSumFaces(	float *coords,
                                    float *assigned_params,
                                    int num_atoms, 
                                    float *volume,
                                    int box_size,
                                    float res,
                                    float ion_size, float wat_size, float asigma,
                                    uint d);

void gpu_computeSumCells(	float *coords,
                            float *assigned_params,
                            int num_atoms, 
                            float *volume,
                            int box_size,
                            float res);

void gpu_computePhi(    float *Q, float *Eps, float *Phi, size_t box_size, 
                        float res, float kappa02);