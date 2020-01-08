#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include "cBackboneProteinCUDAKernels.h"

static void DO_CHECK(cudaError_t res, const char *fn) {
	if (res != cudaSuccess) {
		fprintf(stderr, "Error %s in %s\n", cudaGetErrorString(res), fn);
		exit(1);
	}
}

#define CHECK(fn) DO_CHECK(fn, #fn)

template<typename T> T *to_gpu(T *a, uint size) {
	T *res;
	CHECK(cudaMalloc(&res, size*sizeof(T)));
	CHECK(cudaMemcpy(res, a, size*sizeof(T), cudaMemcpyHostToDevice));
	return res;
}

template<typename T> T *zeros_gpu(uint size, T *cpu_pointer){
	cpu_pointer = new T[size];
	memset(cpu_pointer, 0, size);
	T *gpu_pointer = to_gpu<float>(cpu_pointer, size);
	return gpu_pointer;
}

static bool close(float f1, float f2) {
	return abs(f1 - f2) <= (1e-08f + (1e-05f * abs(f2)));
}

#define NLOOPS 100

int main(void) {
	cudaSetDevice(1);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 1);
	printf("Device: %s\n", prop.name);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int num_aa = 450;
	int num_atoms = 3*num_aa;
	int num_angles = 3*num_aa;
	int batch_size = 64;

	float *cpu_input_angles, *cpu_output_coords, *cpu_dr_dangle, *cpu_A, *cpu_params;
	float *gpu_input_angles = zeros_gpu<float>(batch_size*num_angles, cpu_input_angles);
	float *gpu_output_coords = zeros_gpu<float>(batch_size*num_atoms*3, cpu_output_coords);
	float *gpu_A = zeros_gpu<float>(batch_size*num_atoms*16, cpu_A);
	float *gpu_param = zeros_gpu<float>(6, cpu_params);
	
	int *angles_length = new int[batch_size];
	for(int i=0; i<batch_size; i++)
		angles_length[i] = num_aa;
	int *gpu_angles_length = to_gpu<int>(angles_length, batch_size);

	float *cpu_gradInput, *cpu_gradOutput;
	float *gpu_gradInput = zeros_gpu<float>(batch_size*num_angles, cpu_gradInput);
	float *gpu_gradOutput = zeros_gpu<float>(batch_size*num_atoms, cpu_gradOutput);
	float *gpu_dr_dangle = zeros_gpu<float>(batch_size*num_atoms*num_angles*3, cpu_dr_dangle);

	float *cpu_dangle, *cpu_dr;
	float *gpu_dr = zeros_gpu<float>(batch_size*num_atoms*3, cpu_dr);
	float *gpu_dangle = zeros_gpu<float>(batch_size*num_angles, cpu_dr);
	
	timespec time1, time2;

	CHECK(cudaEventRecord(start));
	for(int i=0; i<NLOOPS; i++){
		gpu_computeCoordinatesBackbone<float>(gpu_input_angles, gpu_param, gpu_output_coords, gpu_A, gpu_angles_length, batch_size, num_aa);
	}
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	float milliseconds = 0;
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("Forward %.3f ms\n", double(milliseconds)/double(NLOOPS));


	cudaEventRecord(start);
	gpu_computeDerivativesBackbone<float>(gpu_input_angles, gpu_param, gpu_dr_dangle, gpu_A, gpu_angles_length, batch_size, num_aa);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Backward grad %0.3f ms\n", milliseconds);

	cudaEventRecord(start);
	gpu_backwardFromCoordsBackbone<float>(gpu_dangle, gpu_dr, gpu_dr_dangle, gpu_angles_length, batch_size, num_aa);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Backward reduce %0.3f ms\n", milliseconds);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
