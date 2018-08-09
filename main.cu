#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "cnpy.h"
#include "cBackboneProteinCUDAKernels.h"

static void DO_CHECK(cudaError_t res, const char *fn) {
  if (res != cudaSuccess) {
    fprintf(stderr, "Error in %s\n", fn);
    exit(1);
  }
}

#define CHECK(fn) DO_CHECK(fn, #fn)


template<typename T> T *to_gpu(cnpy::NpyArray &a) {
  T *res;
  CHECK(cudaMalloc(&res, a.num_bytes()));
  CHECK(cudaMemcpy(res, a.data<T>(), a.num_bytes(), cudaMemcpyHostToDevice));
  return res;
}

template<typename T> T *load_to_gpu(const char *fname) {
  cnpy::NpyArray tmp = cnpy::npy_load(fname);
  return to_gpu<T>(tmp);
}

static bool close(float f1, float f2) {
  return abs(f1 - f2) <= (1e-08f + (1e-05f * abs(f2)));
}

double diff(timespec start, timespec end)
{
  timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return (double)temp.tv_sec + temp.tv_nsec/1000000000.0;
}

#define NLOOPS 100

int main(void) {
  cnpy::NpyArray refGradInput = cnpy::npy_load("gradInputRef.npy");
  float *cpuRefGradInput = (float *)malloc(refGradInput.num_bytes());

  float *gradInput = load_to_gpu<float>("gradInput.npy");
  float *gradOutput = load_to_gpu<float>("gradOutput.npy");
  float *dr_dangle = load_to_gpu<float>("dr_dangle.npy");
  float *input_angles = load_to_gpu<float>("input_angles.npy");
  float *A = load_to_gpu<float>("A.npy");
  int *angles_length = load_to_gpu<int>("angles_length.npy");

  timespec time1, time2;

  cpu_computeDerivativesBackbone(input_angles, dr_dangle, A, angles_length, 64, 650);
  
  cpu_backwardFromCoordsBackbone(gradInput, gradOutput, dr_dangle, angles_length, 64, 650, false);

  CHECK(cudaMemcpy(cpuRefGradInput, gradInput, refGradInput.num_bytes(), cudaMemcpyDeviceToHost));
  for (int i = 0; i < refGradInput.num_vals; i++) {
    if (!close(cpuRefGradInput[i], refGradInput.data<float>()[i])) {
      printf("[%d]: %12.6f, %12.6f\n", i, cpuRefGradInput[i],
             refGradInput.data<float>()[i]);
    }
  }

  CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &time1);

  for (int i = 0; i < NLOOPS; i++) {
    cpu_backwardFromCoordsBackbone(gradInput, gradOutput, dr_dangle, angles_length, 64, 650, false);
  }
  CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &time2);

  printf("Took %0.6fs\n", diff(time1, time2)/NLOOPS);
}
