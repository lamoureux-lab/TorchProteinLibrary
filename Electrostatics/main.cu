//#include <torch/torch.h>
#include<iostream>
#include<cuda.h>

void __global__ fill_eps(float *data, int size, float resolution){
    float x, y, z;
    int flat_idx = 0;
    for(int i=0; i<size; i++){
        x = float(i - size/2) * resolution;
        for(int j=0; j<size; j++){
            y = float(j - size/2) * resolution;
            for(int k=0; k<size; k++){
                z = float(k - size/2) * resolution;
                data[flat_idx] = 0.0;
                flat_idx++;
            }
        }
    }
    return;
}

void __global__ fill_charge(float *data, int size, float resolution){
    int i,j,k;
    float x, y, z;
    int flat_idx = 0;
    for(i=0; i<size; i++){
        x = float(i - size/2) * resolution;
        for(j=0; j<size; j++){
            y = float(j - size/2) * resolution;
            for(k=0; k<size; k++){
                z = float(k - size/2) * resolution;
                data[flat_idx] = 0.0;
                flat_idx++;
            }
        }
    }  
    i = size/2;
    j = size/2;
    k = size/2;
    flat_idx = i*size*size + j*size + k;
    data[flat_idx] = 1.0;
    return;
}

void __global__ check_phi(float *data, int size, float resolution, float* device_error){
    int i,j,k;
    float x0, y0, z0, x, y, z;
    int flat_idx = 0;
    
    *device_error = 0.0; // global variable on the GPU
    float r;
    x0 = (size/2) * resolution;
    y0 = (size/2) * resolution;
    z0 = (size/2) * resolution;
    
    for(i=0; i<size; i++){
        x = float(i - size/2) * resolution;
        for(j=0; j<size; j++){
            y = float(j - size/2) * resolution;
            for(k=0; k<size; k++){
                z = float(k - size/2) * resolution;
                r = sqrt((x0 - x)*(x0 - x) + (y0 - y)*(y0 - y) + (z0 - z)*(z0 - z));
                //printf("%f\n",r);
                if(r>1e-5){
                    //Solution depends on the unit system (eps_0)
                    *device_error += fabs(data[flat_idx] - 4.0*3.14/r);
                    // printf("%f\n",device_error);
                }
                flat_idx++;
            }
        }
    }

}

void __global__ phi_update16x16(float* device_phi1, float* device_phi2, int alpha, 
                    const int N_x, const int N_y, const int N_z){


   #define BDIMX 16
   #define BDIMY 16

   __shared__ float slice[BDIMX+2][BDIMY+2];

   int ix = blockIdx.x*blockDim.x + threadIdx.x;
   int iy = blockIdx.y*blockDim.y + threadIdx.y;

   int tx = threadIdx.x + 1;
   int ty = threadIdx.y + 1;

   int stride = N_x*N_y;
   int i2d = iy*N_x + ix;
   int o2d = 0;
   bool compute_if = ix > 0 && ix < (N_x-1) && iy > 0 && iy < (N_y-1);


   float behind;
   float current = device_phi1[i2d]; o2d = i2d; i2d += stride;
   float infront = device_phi1[i2d]; i2d += stride;

   for(int i=1; i<N_z-1; i++){

    // These go in registers:
    behind = current;
    current= infront;
    infront= device_phi1[i2d];

    i2d += stride;
    o2d += stride;
    __syncthreads();

    // Shared memory

     if (compute_if){
        if(threadIdx.x == 0){ // Halo left
            slice[ty][tx-1]     =   device_phi1[o2d - 1];
        }

        if(threadIdx.x == BDIMX-1){ // Halo right
            slice[ty][tx+1]     =   device_phi1[o2d + 1];
        }

        if(threadIdx.y == 0){ // Halo bottom
            slice[ty-1][tx]     =   device_phi1[o2d - N_x];
        }

        if(threadIdx.y == BDIMY-1){ // Halo top
            slice[ty+1][tx]     =   device_phi1[o2d + N_x];
        }
    }

    __syncthreads();

    slice[ty][tx] = current;

    __syncthreads();

    if (compute_if){
        
        device_phi2[o2d]  = current + (alpha)*(
                         slice[ty][tx-1]+slice[ty][tx]
                        +slice[ty][tx+1]+slice[ty-1][tx]
                        +slice[ty+1][tx]+ behind + infront);
    }

    __syncthreads();

}

}
int main(void){

    float *device_eps, *device_charge, *device_phi1, *device_phi2, *host_phi1;
    int size = 30;
    float resolution = 0.5;
    float *device_error;
    float error;

    cudaMalloc((void**)&device_error, sizeof(float));
    cudaMalloc((void**)&device_eps, size*size*size*sizeof(float));
    cudaMalloc((void**)&device_charge, size*size*size*sizeof(float));
    //cudaMalloc((void**)&device_phi, size*size*size*sizeof(float));

    //Filling dielectric permetivity and charge
    //We don't care about execution time here, so everything
    //is one-threaded on the gpu
    fill_eps<<<1, 1>>>(device_eps, size, resolution);
    fill_charge<<<1, 1>>>(device_charge, size, resolution);
    
    //Solution here
   
    
    // Allocate memory and intialise potentials in host    
    int ary_size = size * size * size * sizeof(float);   
    host_phi1 = (float *)malloc(ary_size);
    fill_eps<<<1, 1>>>(host_phi1, size, resolution);

    // Allocate memory on device and copy from host
    cudaMalloc((void**)&device_phi1, ary_size);
    cudaMalloc((void**)&device_phi2, ary_size);
    cudaMemcpy((void *)device_phi1, (void *)host_phi1, ary_size,
                cudaMemcpyHostToDevice);
    cudaMemcpy((void *)device_phi2, (void *)host_phi1, ary_size,
                cudaMemcpyHostToDevice);
    int alpha=3; 
    // Launch configuration:
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(size/16, size/16, 1);
    phi_update16x16<<<dimGrid, dimBlock>>>(device_phi1, device_phi2, alpha, size, size, size);
    cudaThreadSynchronize();
    
    //The final potential should be in device_phi
    //Checking the result
    check_phi<<<1, 1>>>(device_phi2, size, resolution, device_error);
    cudaMemcpy(&error, device_error, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout<<"Error = "<<error<<std::endl;
    
    //Cleanup
    cudaFree(device_error);
    cudaFree(device_eps);
    cudaFree(device_charge);
    cudaFree(device_phi);
    


    return 0;
}
