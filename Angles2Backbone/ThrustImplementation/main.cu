#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include "cBackboneProteinCUDAKernels.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include <cmath>
#include <iostream>


static void DO_CHECK(cudaError_t res, const char *fn) {
	if (res != cudaSuccess) {
		fprintf(stderr, "Error %s in %s\n", cudaGetErrorString(res), fn);
		exit(1);
	}
}

#define CHECK(fn) DO_CHECK(fn, #fn)

#define NLOOPS 100

struct Vector3
{
	float v[3];
};

struct Matrix44
{
	float d_m1[16];
	__device__ void mat44Mul(const float *d_m2, float *dst) const{

		dst[0] = d_m1[0]*d_m2[0] + d_m1[1]*d_m2[4] + d_m1[2]*d_m2[8] + d_m1[3]*d_m2[12];
		dst[1] = d_m1[0]*d_m2[1] + d_m1[1]*d_m2[5] + d_m1[2]*d_m2[9] + d_m1[3]*d_m2[13];
		dst[2] = d_m1[0]*d_m2[2] + d_m1[1]*d_m2[6] + d_m1[2]*d_m2[10] + d_m1[3]*d_m2[14];
		dst[3] = d_m1[0]*d_m2[3] + d_m1[1]*d_m2[7] + d_m1[2]*d_m2[11] + d_m1[3]*d_m2[15];

		dst[4] = d_m1[4]*d_m2[0] + d_m1[5]*d_m2[4] + d_m1[6]*d_m2[8] + d_m1[7]*d_m2[12];
		dst[5] = d_m1[4]*d_m2[1] + d_m1[5]*d_m2[5] + d_m1[6]*d_m2[9] + d_m1[7]*d_m2[13];
		dst[6] = d_m1[4]*d_m2[2] + d_m1[5]*d_m2[6] + d_m1[6]*d_m2[10] + d_m1[7]*d_m2[14];
		dst[7] = d_m1[4]*d_m2[3] + d_m1[5]*d_m2[7] + d_m1[6]*d_m2[11] + d_m1[7]*d_m2[15];

		dst[8] = d_m1[8]*d_m2[0] + d_m1[9]*d_m2[4] + d_m1[10]*d_m2[8] + d_m1[11]*d_m2[12];
		dst[9] = d_m1[8]*d_m2[1] + d_m1[9]*d_m2[5] + d_m1[10]*d_m2[9] + d_m1[11]*d_m2[13];
		dst[10] = d_m1[8]*d_m2[2] + d_m1[9]*d_m2[6] + d_m1[10]*d_m2[10] + d_m1[11]*d_m2[14];
		dst[11] = d_m1[8]*d_m2[3] + d_m1[9]*d_m2[7] + d_m1[10]*d_m2[11] + d_m1[11]*d_m2[15];

		dst[12] = d_m1[12]*d_m2[0] + d_m1[13]*d_m2[4] + d_m1[14]*d_m2[8] + d_m1[15]*d_m2[12];
		dst[13] = d_m1[12]*d_m2[1] + d_m1[13]*d_m2[5] + d_m1[14]*d_m2[9] + d_m1[15]*d_m2[13];
		dst[14] = d_m1[12]*d_m2[2] + d_m1[13]*d_m2[6] + d_m1[14]*d_m2[10] + d_m1[15]*d_m2[14];
		dst[15] = d_m1[12]*d_m2[3] + d_m1[13]*d_m2[7] + d_m1[14]*d_m2[11] + d_m1[15]*d_m2[15];
	}
	__device__ Vector3 mat44Zero3Mul() const{
		Vector3 dst;
		dst.v[0] = d_m1[3];
		dst.v[1] = d_m1[7];
		dst.v[2] = d_m1[11];
		return dst;
	}
};



struct AMatFunctor
{
	__device__ Matrix44 operator()(const Matrix44& Aim1, Matrix44& Ai) const
	{
		Matrix44 res;
		Aim1.mat44Mul(Ai.d_m1, res.d_m1);
		return res;
	}
};

struct CoordsFunctor
{
    __device__ Vector3 operator()(const Matrix44& Ai)
    {
		Vector3 dst;
		dst.v[0] = Ai.d_m1[3];
		dst.v[1] = Ai.d_m1[7];
		dst.v[2] = Ai.d_m1[11];
        return dst;
    }
};


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
	int num_coords = num_atoms*3;
	int num_angles = 3*num_aa;
	int batch_size = 64;

	thrust::device_vector<float> gpu_input_angles(batch_size*num_angles, 0.0);
	thrust::device_vector<float> gpu_output_coords(batch_size*num_coords, 0.0);
	thrust::device_vector<float> gpu_A(batch_size*num_atoms*16, 0.0);
	thrust::device_vector<float> gpu_param(6, 1.0);
	thrust::device_vector<int> gpu_length(batch_size, num_aa);
	
	Matrix44* raw_mat_ptr = reinterpret_cast<Matrix44*>(thrust::raw_pointer_cast(gpu_A.data()));
	thrust::device_ptr<Matrix44> dev_mat_ptr(raw_mat_ptr);
	thrust::device_vector<Matrix44> gpu_A_mat(dev_mat_ptr, dev_mat_ptr+batch_size*num_atoms);

	Vector3* raw_vec_ptr = reinterpret_cast<Vector3*>(thrust::raw_pointer_cast(gpu_output_coords.data()));
	thrust::device_ptr<Vector3> dev_vec_ptr(raw_vec_ptr);
	thrust::device_vector<Vector3> gpu_output_vec(dev_vec_ptr, dev_vec_ptr+batch_size*num_atoms);
	
	CHECK(cudaEventRecord(start));
	for(int i=0; i<NLOOPS; i++){
		gpu_computeBMatBackbone(thrust::raw_pointer_cast(gpu_input_angles.data()), 
								thrust::raw_pointer_cast(gpu_param.data()),
								thrust::raw_pointer_cast(gpu_A.data()),
								thrust::raw_pointer_cast(gpu_length.data()), 
								batch_size, num_aa);

		thrust::inclusive_scan(gpu_A_mat.begin(), gpu_A_mat.end(), gpu_A_mat.begin(), AMatFunctor());
		thrust::transform(gpu_A_mat.begin(), gpu_A_mat.end(), gpu_output_vec.begin(), CoordsFunctor());
	}
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	float milliseconds = 0;
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("Forward opt %.3f ms\n", double(milliseconds)/double(NLOOPS));
	

	thrust::device_vector<float> gpu_output_coords_old(batch_size*num_coords, 0.0);
	CHECK(cudaEventRecord(start));
	for(int i=0; i<NLOOPS; i++){
		gpu_computeCoordinatesBackbone<float>(	thrust::raw_pointer_cast(gpu_input_angles.data()), 
												thrust::raw_pointer_cast(gpu_param.data()),
												thrust::raw_pointer_cast(gpu_output_coords_old.data()),
												thrust::raw_pointer_cast(gpu_A.data()),
												thrust::raw_pointer_cast(gpu_length.data()), 
												batch_size, num_aa);
	}
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("Forward ref %.3f ms\n", double(milliseconds)/double(NLOOPS));


	thrust::device_vector<float> gpu_output_diff(batch_size*num_coords, 0.0);
	thrust::transform(	gpu_output_coords.begin(), gpu_output_coords.end(), 
						gpu_output_coords_old.begin(),
						gpu_output_diff.begin(),
						thrust::minus<float>());

	float err = std::sqrt(thrust::transform_reduce(gpu_output_diff.begin(), gpu_output_diff.end(), thrust::square<float>(), float(0.0), thrust::plus<float>()));
	printf("Error: %f\n", err);

	// float *cpu_gradInput, *cpu_gradOutput;
	// float *gpu_gradInput = zeros_gpu<float>(batch_size*num_angles, cpu_gradInput);
	// float *gpu_gradOutput = zeros_gpu<float>(batch_size*num_atoms, cpu_gradOutput);
	// float *gpu_dr_dangle = zeros_gpu<float>(batch_size*num_atoms*num_angles*3, cpu_dr_dangle);

	// float *cpu_dangle, *cpu_dr;
	// float *gpu_dr = zeros_gpu<float>(batch_size*num_atoms*3, cpu_dr);
	// float *gpu_dangle = zeros_gpu<float>(batch_size*num_angles, cpu_dr);
	
	timespec time1, time2;

	


	// cudaEventRecord(start);
	// gpu_computeDerivativesBackbone<float>(gpu_input_angles, gpu_param, gpu_dr_dangle, gpu_A, gpu_angles_length, batch_size, num_aa);
	// cudaEventRecord(stop);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&milliseconds, start, stop);
	// printf("Backward grad %0.3f ms\n", milliseconds);

	// cudaEventRecord(start);
	// gpu_backwardFromCoordsBackbone<float>(gpu_dangle, gpu_dr, gpu_dr_dangle, gpu_angles_length, batch_size, num_aa);
	// cudaEventRecord(stop);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&milliseconds, start, stop);
	// printf("Backward reduce %0.3f ms\n", milliseconds);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
