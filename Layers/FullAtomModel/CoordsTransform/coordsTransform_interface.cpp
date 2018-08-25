#include <TH/TH.h>
#include "cConformation.h"
#include <iostream>
#include <string>
#include "nUtil.h"

extern "C" {
    void CoordsTranslate_forward(   THDoubleTensor *input_coords, 
                                    THDoubleTensor *output_coords,
                                    THDoubleTensor *T,
                                    THIntTensor *num_atoms
                                    ){
        if(input_coords->nDimension == 2){
            int batch_size = input_coords->size[0];
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THDoubleTensor *single_input_coords = THDoubleTensor_new();
                THDoubleTensor *single_output_coords = THDoubleTensor_new();
                THDoubleTensor_select(single_input_coords, input_coords, 0, i);
                THDoubleTensor_select(single_output_coords, output_coords, 0, i);
                
                cVector3 translation(THDoubleTensor_get2d(T, i, 0), THDoubleTensor_get2d(T, i, 1), THDoubleTensor_get2d(T, i, 2));
                ProtUtil::translate( single_input_coords, translation, single_output_coords, THIntTensor_get1d(num_atoms, i));

                THDoubleTensor_free(single_input_coords);
                THDoubleTensor_free(single_output_coords);
            }
        }else{
            std::cout<<"Not implemented\n";
        }                             
    }
    void CoordsTranslate_backward(  THDoubleTensor *grad_output_coords, 
                                    THDoubleTensor *grad_input_coords
                                    ){
        if(grad_output_coords->nDimension == 2){
            THDoubleTensor_copy(grad_input_coords, grad_output_coords);
        }else{
            std::cout<<"Not implemented\n";
        }
    }
    void CoordsRotate_forward(  THDoubleTensor *input_coords, 
                                THDoubleTensor *output_coords,
                                THDoubleTensor *R,
                                THIntTensor *num_atoms
                                ){
        if(input_coords->nDimension == 2){
            int batch_size = input_coords->size[0];
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THDoubleTensor *single_input_coords = THDoubleTensor_new();
                THDoubleTensor *single_output_coords = THDoubleTensor_new();
                THDoubleTensor *single_R = THDoubleTensor_new();
                THDoubleTensor_select(single_input_coords, input_coords, 0, i);
                THDoubleTensor_select(single_output_coords, output_coords, 0, i);
                THDoubleTensor_select(single_R, R, 0, i);
                
                cMatrix33 _R = ProtUtil::tensor2Matrix33(single_R);
                ProtUtil::rotate(single_input_coords, _R, single_output_coords, THIntTensor_get1d(num_atoms, i));

                THDoubleTensor_free(single_input_coords);
                THDoubleTensor_free(single_output_coords);
            }
        }else{
            std::cout<<"Not implemented\n";
        }                             
    }
    void CoordsRotate_backward( THDoubleTensor *grad_output_coords, 
                                THDoubleTensor *grad_input_coords,
                                THDoubleTensor *R,
                                THIntTensor *num_atoms){
        
        if(grad_output_coords->nDimension == 2){
            int batch_size = grad_output_coords->size[0];

            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THDoubleTensor *single_grad_output_coords = THDoubleTensor_new();
                THDoubleTensor *single_grad_input_coords = THDoubleTensor_new();
                THDoubleTensor *single_R = THDoubleTensor_new();
                THDoubleTensor_select(single_grad_output_coords, grad_output_coords, 0, i);
                THDoubleTensor_select(single_grad_input_coords, grad_input_coords, 0, i);
                THDoubleTensor_select(single_R, R, 0, i);

                cMatrix33 _R = ProtUtil::tensor2Matrix33(single_R);
                _R = _R.getTranspose();
                ProtUtil::rotate(single_grad_output_coords, _R, single_grad_input_coords, THIntTensor_get1d(num_atoms, i));
                
                THDoubleTensor_free(single_R);
                THDoubleTensor_free(single_grad_input_coords);
                THDoubleTensor_free(single_grad_output_coords);
            }

        }else{
            std::cout<<"Not implemented\n";
        }
    }
    void getBBox( THDoubleTensor *input_coords,
                  THDoubleTensor *a, THDoubleTensor *b,
                  THIntTensor *num_atoms){

        if(input_coords->nDimension == 2){
            int batch_size = input_coords->size[0];
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THDoubleTensor *single_input_coords = THDoubleTensor_new();
                THDoubleTensor *single_a = THDoubleTensor_new();
                THDoubleTensor *single_b = THDoubleTensor_new();
                THDoubleTensor_select(single_input_coords, input_coords, 0, i);
                THDoubleTensor_select(single_a, a, 0, i);
                THDoubleTensor_select(single_b, b, 0, i);
                
                cVector3 b0, b1;
                ProtUtil::computeBoundingBox(single_input_coords, THIntTensor_get1d(num_atoms, i), b0, b1);
                THDoubleTensor_set1d(single_a, 0, b0.v[0]);
                THDoubleTensor_set1d(single_a, 1, b0.v[1]);
                THDoubleTensor_set1d(single_a, 2, b0.v[2]);

                THDoubleTensor_set1d(single_b, 0, b1.v[0]);
                THDoubleTensor_set1d(single_b, 1, b1.v[1]);
                THDoubleTensor_set1d(single_b, 2, b1.v[2]);

                THDoubleTensor_free(single_input_coords);
                THDoubleTensor_free(single_a);
                THDoubleTensor_free(single_b);
            }

        }else{
            std::cout<<"Not implemented\n";
        }
    }
    void getRandomRotation( THDoubleTensor *R ){
        THGenerator *gen = THGenerator_new();
 		THRandom_seed(gen);
        
        if(R->nDimension == 3){
            int batch_size = R->size[0];
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THDoubleTensor *single_R = THDoubleTensor_new();
                THDoubleTensor_select(single_R, R, 0, i);
                
                cMatrix33 _R = ProtUtil::getRandomRotation(gen);
                ProtUtil::matrix2Tensor(_R, single_R);                
                
                THDoubleTensor_free(single_R);
            }

        }else{
            std::cout<<"Not implemented\n";
        }
        THGenerator_free(gen);
    }
    void getRandomTranslation( THDoubleTensor *T, THDoubleTensor *a, THDoubleTensor *b, int volume_size ){
        THGenerator *gen = THGenerator_new();
 		THRandom_seed(gen);
        
        if(T->nDimension == 2){
            int batch_size = T->size[0];
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THDoubleTensor *single_T = THDoubleTensor_new();
                THDoubleTensor_select(single_T, T, 0, i);
                
                cVector3 _a(THDoubleTensor_get2d(a, i, 0), THDoubleTensor_get2d(a, i, 1), THDoubleTensor_get2d(a, i, 2));
                cVector3 _b(THDoubleTensor_get2d(b, i, 0), THDoubleTensor_get2d(b, i, 1), THDoubleTensor_get2d(b, i, 2));

                cVector3 _T = ProtUtil::getRandomTranslation(gen, volume_size, _a, _b);
                THDoubleTensor_set1d(single_T, 0, _T.v[0]);
                THDoubleTensor_set1d(single_T, 1, _T.v[1]);
                THDoubleTensor_set1d(single_T, 2, _T.v[2]);

                THDoubleTensor_free(single_T);
            }

        }else{
            std::cout<<"Not implemented\n";
        }
        THGenerator_free(gen);
    }
}