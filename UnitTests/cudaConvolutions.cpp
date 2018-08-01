#include <TH.h>
#include <THC.h>
#include <iostream>
#include <cufft.h>
#include <math.h>

#define CUDA_REAL_TENSOR_VAR THCudaTensor
#define CUDA_REAL_TENSOR(X) THCudaTensor_##X

void fill_input1d(THCState *state, CUDA_REAL_TENSOR_VAR *tensor){
    int k0 = 5.0;
    for(int i=0;i<tensor->size[0];i++){
        for(int j=0; j<tensor->size[1]; j++){
            float var = cos(2.0*M_PI*k0*j/tensor->size[1]);
            CUDA_REAL_TENSOR(set2d)(state, tensor, i, j, var);
        }
    }
}


void fill_input3d(THCState *state, CUDA_REAL_TENSOR_VAR *tensor){
    float k0 = .5, k1 = .5, k2 = .5;
    for(int i=0;i<tensor->size[0];i++){
        for(int j=0; j<tensor->size[1]; j++){
            for(int k=0; k<tensor->size[2]; k++){
                for(int l=0; l<tensor->size[3]; l++){
                    float var = cos(2.0*M_PI*(k0*j/tensor->size[1] + k1*k/tensor->size[2] + k2*l/tensor->size[2]));
                    CUDA_REAL_TENSOR(set4d)(state, tensor, i, j, k, l, var);
                }
            }
        }
    }
}

void cmp_input1d(THCState *state, CUDA_REAL_TENSOR_VAR *tensor){
    int k0 = 5.0;
    float error = 0.0;
    for(int i=0;i<tensor->size[0];i++){
        for(int j=0; j<tensor->size[1]; j++){
            float var = cos(2.0*M_PI*k0*j/tensor->size[1]);
            error += fabs(var - CUDA_REAL_TENSOR(get2d)(state, tensor, i, j));
        }
    }
    std::cout<<"Error: "<<error<<std::endl;
}

void cmp_input3d(THCState *state, CUDA_REAL_TENSOR_VAR *tensor){
    float k0 = .5, k1 = .5, k2 = .5;
    float error = 0.0;
    for(int i=0;i<tensor->size[0];i++){
        for(int j=0; j<tensor->size[1]; j++){
            for(int k=0; k<tensor->size[2]; k++){
                for(int l=0; l<tensor->size[3]; l++){
                    float var = cos(2.0*M_PI*(k0*j/tensor->size[1] + k1*k/tensor->size[2] + k2*l/tensor->size[2]));
                    error += fabs(var - CUDA_REAL_TENSOR(get4d)(state, tensor, i, j, k, l));
                }
            }
        }
    }
    std::cout<<"Error: "<<error<<std::endl;
}

void print_output_real1d(THCState *state, CUDA_REAL_TENSOR_VAR *tensor){
    int k0 = 2.0;
    for(int i=0;i<tensor->size[0];i++){
        std::cout<<"Batch "<<i<<"\n";
        for(int j=0; j<tensor->size[1]; j++){
            std::cout<<CUDA_REAL_TENSOR(get2d)(state, tensor, i, j)<<", ";
        }
        std::cout<<std::endl;
    }
}

void print_output_real3d(THCState *state, CUDA_REAL_TENSOR_VAR *tensor){
    
    for(int i=0;i<tensor->size[0];i++){
        std::cout<<"Batch "<<i<<"\n";
        for(int j=0; j<tensor->size[1]; j++){
            for(int k=0; k<tensor->size[2]; k++){
                for(int l=0; l<tensor->size[3]; l++){
                    float amp = CUDA_REAL_TENSOR(get4d)(state, tensor, i, j, k, l);
                    std::cout<<j<<" "<<k<<" "<<l<<" "<<amp<<"\n";
                }
            }
        }
        std::cout<<std::endl;
    }
}

void print_output_complex1d(THCState *state, CUDA_REAL_TENSOR_VAR *tensor){
    
    for(int i=0;i<tensor->size[0];i++){
        std::cout<<"Batch "<<i<<"\n";
        for(int j=0; j<tensor->size[1]; j+=2){
            float amp = sqrt(CUDA_REAL_TENSOR(get2d)(state, tensor, i, j)*CUDA_REAL_TENSOR(get2d)(state, tensor, i, j)\
             - CUDA_REAL_TENSOR(get2d)(state, tensor, i, j+1)*CUDA_REAL_TENSOR(get2d)(state, tensor, i, j+1))/16;
            std::cout<<amp<<", ";
        }
        std::cout<<std::endl;
    }
}

void print_output_complex3d(THCState *state, CUDA_REAL_TENSOR_VAR *tensor){
    
    for(int i=0;i<tensor->size[0];i++){
        std::cout<<"Batch "<<i<<"\n";
        for(int j=0; j<tensor->size[1]; j++){
            for(int k=0; k<tensor->size[2]; k++){
                for(int l=0; l<tensor->size[3]; l+=2){
                    float amp = sqrt(CUDA_REAL_TENSOR(get4d)(state, tensor, i, j, k, l)*CUDA_REAL_TENSOR(get4d)(state, tensor, i, j, k, l)\
                    + CUDA_REAL_TENSOR(get4d)(state, tensor, i, j, k, l+1)*CUDA_REAL_TENSOR(get4d)(state, tensor, i, j, k, l+1));
                    std::cout<<j<<" "<<k<<" "<<l<<" "<<amp<<"\n";
                }
            }
        }
        std::cout<<std::endl;
    }
}

void test1d(THCState* state){
    int batch_size = 4;
    int in_size = 32;
    int out_size = in_size/2 + 1;
    
    CUDA_REAL_TENSOR_VAR *input = CUDA_REAL_TENSOR(newWithSize2d)(state, batch_size, in_size);
    CUDA_REAL_TENSOR_VAR *output = CUDA_REAL_TENSOR(newWithSize2d)(state, batch_size, out_size*2);
    CUDA_REAL_TENSOR_VAR *input_norm = CUDA_REAL_TENSOR(newWithSize2d)(state, batch_size, out_size*2);
    fill_input1d(state, input);
    print_output_real1d(state, input);

    cufftHandle plan_fwd, plan_bwd;
        
    int rank = 1;
    int dimensions_in[] = {in_size};
    int dimensions_out[] = {out_size};
    int istride = 1, ostride = 1;   
    int iembed[] = {0};
    int oembed[] = {0};
    cufftPlanMany(  &plan_fwd, rank, dimensions_in, 
                    iembed, istride, in_size, 
                    oembed, ostride, out_size,
                    CUFFT_R2C, batch_size);

    cufftPlanMany(  &plan_bwd, rank, dimensions_in, 
                    oembed, ostride, out_size,
                    iembed, istride, in_size, 
                    CUFFT_C2R, batch_size);


    cufftExecR2C(plan_fwd, CUDA_REAL_TENSOR(data)(state,input), (cufftComplex*)CUDA_REAL_TENSOR(data)(state,output));
    print_output_complex1d(state, output);

    cufftExecC2R(plan_bwd, (cufftComplex*)CUDA_REAL_TENSOR(data)(state,output), CUDA_REAL_TENSOR(data)(state,input));

    CUDA_REAL_TENSOR(div)(state, input_norm, input, in_size);
    print_output_real1d(state, input_norm);
    cmp_input1d(state, input_norm);
    cufftDestroy(plan_fwd);
    
    
    CUDA_REAL_TENSOR(free)(state, input);
    CUDA_REAL_TENSOR(free)(state, output);
}

void test3d(THCState* state){
    int batch_size = 1;
    int in_size = 8;
    int out_size = in_size/2 + 1;
    
    CUDA_REAL_TENSOR_VAR *input = CUDA_REAL_TENSOR(newWithSize4d)(state, batch_size, in_size, in_size, in_size);
    CUDA_REAL_TENSOR_VAR *output = CUDA_REAL_TENSOR(newWithSize4d)(state, batch_size, in_size, in_size, out_size*2);
    CUDA_REAL_TENSOR_VAR *input_norm = CUDA_REAL_TENSOR(newWithSize4d)(state, batch_size, in_size, in_size, out_size*2);
    fill_input3d(state, input);
    print_output_real3d(state, input);
    cufftHandle plan_fwd, plan_bwd;
        
    int rank = 3;
    int dimensions_in[] = {in_size, in_size, in_size};
    int dimensions_out[] = {in_size, in_size, out_size};
    int istride = 1, ostride = 1;   
    int iembed[] = {in_size, in_size, in_size};
    int oembed[] = {in_size, in_size, out_size};
    cufftPlanMany(  &plan_fwd, rank, dimensions_in, 
                    iembed, istride, in_size*in_size*in_size, 
                    oembed, ostride, out_size*in_size*in_size,
                    CUFFT_R2C, batch_size);

    cufftPlanMany(  &plan_bwd, rank, dimensions_in, 
                    oembed, ostride, out_size*in_size*in_size,
                    iembed, istride, in_size*in_size*in_size, 
                    CUFFT_C2R, batch_size);


    cufftExecR2C(plan_fwd, CUDA_REAL_TENSOR(data)(state, input), (cufftComplex*)CUDA_REAL_TENSOR(data)(state, output));
    // print_output_complex3d(state, output);
    
    cufftExecC2R(plan_bwd, (cufftComplex*)CUDA_REAL_TENSOR(data)(state, output), CUDA_REAL_TENSOR(data)(state, input));

    CUDA_REAL_TENSOR(div)(state, input_norm, input, in_size*in_size*in_size);
    
    cmp_input3d(state, input_norm);
    cufftDestroy(plan_fwd);
    
    
    CUDA_REAL_TENSOR(free)(state, input);
    CUDA_REAL_TENSOR(free)(state, output);
}

int main(int argc, char** argv)
{

    THCState* state = (THCState*) malloc(sizeof(THCState));    
	memset(state, 0, sizeof(THCState));
	THCudaInit(state); 
    
    test3d(state);
    
    THCudaShutdown(state);
	free(state);
	
	return 0;
}