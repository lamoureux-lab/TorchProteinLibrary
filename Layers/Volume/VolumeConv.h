
#define REAL float
// #define REAL double


void cpu_VolumeConv(REAL *d_volume1,  REAL *d_volume2,  REAL *d_output, int batch_size, int volume_size, bool conjugate);
// Computes F(tau) = \int vol1(r)vol2(tau-r) dr
