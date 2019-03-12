#define REAL float

void gpu_VolumeRMSD(REAL *d_volume,  
                    REAL add, 
                    REAL T0_x, REAL T0_y, REAL T0_z,
                    REAL D_x, REAL D_y, REAL D_z, 
                    int volume_size, float resolution);