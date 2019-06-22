//#include <torch/torch.h>
#include<iostream>
#include <cusp/gallery/poisson.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>
#include <cusp/precond/diagonal.h>


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
    int p,q,r,id;
    //Solution here
    cusp::coo_matrix<int, float, cusp::device_memory> A;
    // create a matrix for a Poisson problem on a sizexsizexsize grid
    cusp::gallery::poisson7pt(A, size, size, size);
    int N=size*size*size;
    //filling the coeffecient matrix with eps
    for(p=0; p<size; p++)
      {  for(q=0; q<size; q++)
           {  for(r=0; r<size; r++)
                 {
                  id=p*size*size+q*size+r;
                  
                  A[id][id]=(-6)*device_eps*A[id][id];
                  A[id][id+1]=device_eps[p*size*size+q*size+(r+1)]*A[id][id+1];
                  A[id][id-1]=device_eps[p*size*size+q*size+(r-1)]*A[id][id-1];
                  A[id][id+size]=device_eps[p*size*size+(q+1)*size+(r)]*A[id][id+size];
                  A[id][id-size]=device_eps[p*size*size+(q-1)*size+(r)]*A[id][id-size];
                  A[id][id+2*size]=device_eps[(p+1)*size*size+q*size+(r)]*A[id][id+2*size];
                  A[id][id-2*size]=device_eps[(p-1)*size*size+q*size+(r)]*A[id][id-2*size];
                 }
           }  
      }                       
                  
      cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
      cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);//this is not correct,has to be improved.
      
      cusp::default_monitor<float> monitor(b, 100, 1e-6);
      // setup preconditioner
      cusp::precond::diagonal<float, cusp::device_memory> M(A);
      // solve
      cusp::krylov::cg(A, x, b, monitor, M);
    
                  
                  
        

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
