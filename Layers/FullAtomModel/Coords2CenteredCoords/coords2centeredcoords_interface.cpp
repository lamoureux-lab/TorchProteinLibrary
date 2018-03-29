#include <TH/TH.h>
#include "cConformation.h"
#include <iostream>
#include <string>
#include "nUtil.h"

bool int2bool(int var){
    bool bool_var;
    if(var == 1){
        bool_var = true;
    }else if(var == 0){
        bool_var = false;
    }else{
        std::cout<<"unknown var = "<<var<<std::endl;
        throw std::string("unknown var");
    }
    return bool_var;
}

extern "C" {
    void Coords2CenteredCoords_forward( THDoubleTensor *input_coords, 
                                        THDoubleTensor *output_coords,
                                        int volume_size,
                                        THDoubleTensor *R,
                                        THDoubleTensor *T,
                                        int rotate,
                                        int translate
                                    ){
        bool rot = int2bool(rotate);
        bool trans = int2bool(translate);
        THGenerator *gen = THGenerator_new();
 		THRandom_seed(gen);
        if(input_coords->nDimension == 1){
            
            uint num_atoms = input_coords->size[0]/3;
            THDoubleTensor_copy(output_coords, input_coords);
            cVector3 b0, b1;
            ProtUtil::computeBoundingBox(input_coords, b0, b1);
            cVector3 center_box = (b0 + b1)*0.5;
            ProtUtil::translate( output_coords, -center_box);
            if(rot){
                cMatrix33 _R = ProtUtil::getRandomRotation(gen);
                ProtUtil::matrix2Tensor(_R, R);
                ProtUtil::rotate(output_coords, _R);
            }
            cVector3 center_volume(volume_size/2.0, volume_size/2.0, volume_size/2.0);
            ProtUtil::translate(output_coords, center_volume);
            if(trans){
                cVector3 _T(THDoubleTensor_data(T));
                _T = ProtUtil::getRandomTranslation(gen, volume_size, b0, b1);
                ProtUtil::translate(output_coords, _T);
            }

        }else if(input_coords->nDimension == 2){

            
            int batch_size = input_coords->size[0];
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THDoubleTensor *single_input_coords = THDoubleTensor_new();
                THDoubleTensor *single_output_coords = THDoubleTensor_new();
                THDoubleTensor *single_R = THDoubleTensor_new();
                THDoubleTensor *single_T = THDoubleTensor_new();
                THDoubleTensor_select(single_input_coords, input_coords, 0, i);
                THDoubleTensor_select(single_output_coords, output_coords, 0, i);
                THDoubleTensor_select(single_R, R, 0, i);
                THDoubleTensor_select(single_T, T, 0, i);


                uint num_atoms = single_input_coords->size[0]/3;    
                THDoubleTensor_copy(single_output_coords, single_input_coords);
                cVector3 b0, b1;
                ProtUtil::computeBoundingBox(single_input_coords, b0, b1);
                cVector3 center_box = (b0 + b1)*0.5;
                ProtUtil::translate( single_output_coords, -center_box);
                if(rot){
                    cMatrix33 _R = ProtUtil::getRandomRotation(gen);
                    ProtUtil::matrix2Tensor(_R, single_R);
                    ProtUtil::rotate(single_output_coords, _R);
                }
                cVector3 center_volume(volume_size/2.0, volume_size/2.0, volume_size/2.0);
                ProtUtil::translate(single_output_coords, center_volume);
                if(trans){
                    cVector3 _T(THDoubleTensor_data(single_T));
                    _T = ProtUtil::getRandomTranslation(gen, volume_size, b0, b1);
                    ProtUtil::translate(single_output_coords, _T);
                }

                THDoubleTensor_free(single_R);
                THDoubleTensor_free(single_T);
                THDoubleTensor_free(single_input_coords);
                THDoubleTensor_free(single_output_coords);
            }

        }else{
            std::cout<<"Not implemented\n";
        }
        THGenerator_free(gen);
    }
    void Coords2CenteredCoords_backward(    THDoubleTensor *grad_output_coords, 
                                            THDoubleTensor *grad_input_coords,
                                            THDoubleTensor *R,
                                            THDoubleTensor *T,
                                            int rotate,
                                            int translate
                                        ){
        bool rot = int2bool(rotate);
        bool trans = int2bool(translate);
        if(grad_output_coords->nDimension == 1){
            
            THDoubleTensor_copy(grad_input_coords, grad_output_coords);
            cVector3 fp_trans(THDoubleTensor_data(grad_output_coords));
            ProtUtil::translate( grad_input_coords, -fp_trans);
            if(rot){
                cMatrix33 _R = ProtUtil::tensor2Matrix33(R);
                _R = _R.getTranspose();
                ProtUtil::rotate(grad_input_coords, _R);
            }

        }else if(grad_output_coords->nDimension == 2){
            int batch_size = grad_output_coords->size[0];

            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THDoubleTensor *single_grad_output_coords = THDoubleTensor_new();
                THDoubleTensor *single_grad_input_coords = THDoubleTensor_new();
                THDoubleTensor *single_R = THDoubleTensor_new();
                THDoubleTensor *single_T = THDoubleTensor_new();
                THDoubleTensor_select(single_grad_output_coords, grad_output_coords, 0, i);
                THDoubleTensor_select(single_grad_input_coords, grad_input_coords, 0, i);
                THDoubleTensor_select(single_R, R, 0, i);
                THDoubleTensor_select(single_T, T, 0, i);

                THDoubleTensor_copy(single_grad_output_coords, single_grad_input_coords);
                cVector3 fp_trans(THDoubleTensor_data(single_grad_output_coords));
                ProtUtil::translate( single_grad_input_coords, -fp_trans);
                if(rot){
                    cMatrix33 _R = ProtUtil::tensor2Matrix33(single_R);
                    _R = _R.getTranspose();
                    ProtUtil::rotate(single_grad_input_coords, _R);
                }

                THDoubleTensor_free(single_R);
                THDoubleTensor_free(single_T);
                THDoubleTensor_free(single_grad_input_coords);
                THDoubleTensor_free(single_grad_output_coords);
            }

        }else{
            std::cout<<"Not implemented\n";
        }
    }
    
}