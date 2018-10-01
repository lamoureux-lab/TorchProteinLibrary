
#define REAL float
// #define REAL double


void cpu_VolumeConv(REAL *d_volume1,  REAL *d_volume2,  REAL *d_output, int batch_size, int volume_size);
// Computes F(tau) = \int vol1(r)vol2(tau-r) dr

void cpu_VolumeConvGrad( REAL *d_gradOutput, REAL *d_volume1, REAL *d_volume2, REAL *d_gradVolume1, int batch_size, int volume_size, bool conjugate);
// Computes the following:
// conjugate == true
// F(tau) = vol2(tau) \int grad(r)vol1(r-tau) dr
// conjugate == false
// F(tau) = vol2(tau) \int grad(r)vol1(tau-r) dr