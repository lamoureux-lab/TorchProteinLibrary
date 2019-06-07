//#include <torch/torch.h>
#include<iostream>

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
                    *device_error += fabs(data[flat_idx] - 4.0*M_PI/r);
                    // printf("%f\n",device_error);
                }
                flat_idx++;
            }
        }
    }

}


int main(void){

    float *device_eps, *device_charge, *device_phi;
    int size = 30;
    float resolution = 0.5;
    float *device_error;
    float error;

    cudaMalloc((void**)&device_error, sizeof(float));
    cudaMalloc((void**)&device_eps, size*size*size*sizeof(float));
    cudaMalloc((void**)&device_charge, size*size*size*sizeof(float));
    cudaMalloc((void**)&device_phi, size*size*size*sizeof(float));

    //Filling dielectric permetivity and charge
    //We don't care about execution time here, so everything
    //is one-threaded on the gpu
    fill_eps<<<1, 1>>>(device_eps, size, resolution);
    fill_charge<<<1, 1>>>(device_charge, size, resolution);

    //Solution here



    //The final potential should be in device_phi
    //Checking the result
    check_phi<<<1, 1>>>(device_phi, size, resolution, device_error);
    cudaMemcpy(&error, device_error, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout<<"Error = "<<error<<std::endl;
    
    //Cleanup
    cudaFree(device_error);
    cudaFree(device_eps);
    cudaFree(device_charge);
    cudaFree(device_phi);
    


    return 0;
}