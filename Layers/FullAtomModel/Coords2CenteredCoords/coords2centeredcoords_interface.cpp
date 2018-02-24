#include <TH/TH.h>
#include "cConformation.h"
#include <iostream>
#include <string>
#include "nUtil.h"

extern "C" {
    void Coords2CenteredCoords_forward( THDoubleTensor *input_coords, 
                                        THDoubleTensor *output_coords,
                                        int volume_size,
                                        THDoubleTensor *R,
                                        THDoubleTensor *T
                                    ){
        THGenerator *gen = THGenerator_new();
 		THRandom_seed(gen);
        if(input_coords->nDimension == 1){
            
            uint num_atoms = input_coords->size[0]/3;
            THDoubleTensor_copy(output_coords, input_coords);
            cVector3 b0, b1;
            ProtUtil::computeBoundingBox(input_coords, b0, b1);
            cVector3 center_box = (b0 + b1)*0.5;
            ProtUtil::translate( output_coords, -center_box);
            cMatrix33 _R = ProtUtil::getRandomRotation(gen);
            ProtUtil::matrix2Tensor(_R, R);
            ProtUtil::rotate(output_coords, _R);
            cVector3 center_volume(volume_size/2.0, volume_size/2.0, volume_size/2.0);
            ProtUtil::translate(output_coords, center_volume);
            cVector3 _T(THDoubleTensor_data(T));
            _T = ProtUtil::getRandomTranslation(gen, volume_size, b0, b1);
            ProtUtil::translate(output_coords, _T);
            
        }else{
            std::cout<<"Not implemented\n";
        }
        THGenerator_free(gen);
    }
    void Coords2CenteredCoords_backward(    THDoubleTensor *grad_output_coords, 
                                            THDoubleTensor *grad_input_coords,
                                            THDoubleTensor *R,
                                            THDoubleTensor *T
                                        ){
        if(grad_output_coords->nDimension == 1){
            THDoubleTensor_copy(grad_input_coords, grad_output_coords);
            cVector3 fp_trans(THDoubleTensor_data(grad_output_coords));
            ProtUtil::translate( grad_input_coords, -fp_trans);
            cMatrix33 _R = ProtUtil::tensor2Matrix33(R);
            _R = _R.getTranspose();
            std::cout<<_R<<"\n";
            ProtUtil::rotate(grad_input_coords, _R);
        }else{
            std::cout<<"Not implemented\n";
        }
    }
    
}