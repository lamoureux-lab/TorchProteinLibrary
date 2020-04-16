void gpu_coords2volume(	float *coords,
                        int num_atoms,
                        float *volume,
						int spatial_dim,
                        float res);

void gpu_coords2volume_cell(float *coords,
                            int num_atoms,
                            float *volume,
						    int spatial_dim,
                            float res);