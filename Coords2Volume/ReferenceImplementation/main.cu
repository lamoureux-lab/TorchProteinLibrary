#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include "Kernel.h"
#include "HashKernel.h"

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

template<typename T> T *to_cpu(T *a, uint size) {
	T *res;
	res = (float*)malloc(size*sizeof(T));
	CHECK(cudaMemcpy(res, a, size*sizeof(T), cudaMemcpyDeviceToHost));
	return res;
}

template<typename T> T *zeros_gpu(uint size){
    T *gpu_pointer;
    CHECK(cudaMalloc(&gpu_pointer, size*sizeof(T)));
	CHECK(cudaMemset(gpu_pointer, 0, size*sizeof(T)));
	return gpu_pointer;
}

static bool close(float f1, float f2) {
	return abs(f1 - f2) <= (1e-08f + (1e-05f * abs(f2)));
}

#define NLOOPS 1

int main(void) {
	cudaSetDevice(1);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 1);
	printf("Device: %s\n", prop.name);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int num_atoms = 3000;
    int L=120;

    std::srand(42);
	
	float *cpu_input_coords = new float[3*num_atoms];
    for(int i=0; i<num_atoms; i++){
        for(int k=0; k<3; k++)
            cpu_input_coords[3*i + k] = L*std::rand()/float(RAND_MAX) + L/2;
    }
	float *gpu_input_coords = to_gpu<float>(cpu_input_coords, num_atoms*3);
	float *gpu_output_volume = zeros_gpu<float>(L*L*L);
    	
	timespec time1, time2;

	CHECK(cudaEventRecord(start));
	for(int i=0; i<NLOOPS; i++){
		gpu_coords2volume(gpu_input_coords, num_atoms, gpu_output_volume, L, 1.0);
	}
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	float milliseconds = 0;
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("Forward(ref) %.3f ms\n", double(milliseconds)/double(NLOOPS));

    float *cpu_output_volume_ref = to_cpu<float>(gpu_output_volume, L*L*L);



    CHECK(cudaEventRecord(start));
	for(int i=0; i<NLOOPS; i++){
		gpu_coords2volume_cell(gpu_input_coords, num_atoms, gpu_output_volume, L, 1.0);
	}
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	milliseconds = 0;
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("Forward(cell) %.3f ms\n", double(milliseconds)/double(NLOOPS));

	float *cpu_output_volume_cell = to_cpu<float>(gpu_output_volume, L*L*L);
    float err = 0.0;
    for(int i=0; i<L*L*L; i++){
        err += fabs(cpu_output_volume_ref[i] - cpu_output_volume_cell[i]);
    }
    printf("Error cell: %.3f\n", err);



	CHECK(cudaEventRecord(start));
	for(int i=0; i<NLOOPS; i++){
		gpu_BuildHashTable(gpu_input_coords, num_atoms, gpu_output_volume, L, 1.0, 2);
	}
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	milliseconds = 0;
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("Forward(hash) %.3f ms\n", double(milliseconds)/double(NLOOPS));

    float *cpu_output_volume_hash = to_cpu<float>(gpu_output_volume, L*L*L);
    err = 0.0;
    for(int i=0; i<L*L*L; i++){
        err += fabs(cpu_output_volume_ref[i] - cpu_output_volume_hash[i]);
    }
    printf("Error hash: %.3f\n", err);



	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}