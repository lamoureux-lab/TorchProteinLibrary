
void gpu_computeCoords2Volume(	float *coords,
                                float *assigned_params,
							    int *num_atoms, 
								float *volume,
								int batch_size,
                                int box_size,
                                int coords_stride,
								float res);