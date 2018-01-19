#include <TH/TH.h>
#include "cConformation.h"
#include <iostream>
#include <string>

extern "C" {
    void Angles2Coords_forward(  const char* sequence,
                                THDoubleTensor *input_angles, 
                                THDoubleTensor *output_coords
                            ){
        if(input_angles->nDimension == 2){
            std::string aa(sequence);
            uint length = aa.length();
            std::cout<<length<<"\n";
            THDoubleTensor *dummy_grad = THDoubleTensor_newWithSize2d(input_angles->size[0], input_angles->size[1]);
            cConformation conf( aa, THDoubleTensor_data(input_angles), THDoubleTensor_data(dummy_grad), 
                                length, THDoubleTensor_data(output_coords));
            THDoubleTensor_free(dummy_grad);
        }else{
            std::cout<<"Not implemented\n";
        }
    }
    void Angles2Coords_backward(     THDoubleTensor *grad_atoms,
                                    THDoubleTensor *grad_angles,
                                    const char* sequence,
                                    THDoubleTensor *input_angles
                            ){
        if(input_angles->nDimension == 2){
            std::string aa(sequence);
            uint length = aa.length();
            THDoubleTensor *dummy_coords = THDoubleTensor_newWithSize1d( 3*length*18);
            cConformation conf( aa, THDoubleTensor_data(input_angles), THDoubleTensor_data(grad_angles), 
                                length, THDoubleTensor_data(dummy_coords));
            conf.backward(conf.root, THDoubleTensor_data(grad_atoms));
            THDoubleTensor_free(dummy_coords);
        }else{
            std::cout<<"Not implemented\n";
        }
    }
}