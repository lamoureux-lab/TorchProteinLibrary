#include <TH/TH.h>
#include "cConformation.h"
#include <iostream>
#include <string>
#include "nUtil.h"
bool int2bool(int add_terminal){
    bool add_term;
    if(add_terminal == 1){
        add_term = true;
    }else if(add_terminal == 0){
        add_term = false;
    }else{
        std::cout<<"unknown add_terminal = "<<add_terminal<<std::endl;
        throw std::string("unknown add_terminal");
    }
    return add_term;
}
extern "C" {
    void Angles2Coords_forward(  const char* sequence,
                                THDoubleTensor *input_angles, 
                                THDoubleTensor *output_coords,
                                THByteTensor *res_names,
                                THByteTensor *atom_names,
                                int add_terminal
                            ){
        if(input_angles->nDimension == 2){
            std::string aa(sequence);
            uint length = aa.length();
            bool add_term = int2bool(add_terminal);
            THDoubleTensor *dummy_grad = THDoubleTensor_newWithSize2d(input_angles->size[0], input_angles->size[1]);
            cConformation conf( aa, THDoubleTensor_data(input_angles), THDoubleTensor_data(dummy_grad), 
                                length, THDoubleTensor_data(output_coords));
            
            THByteTensor *single_atom_name = THByteTensor_new();
            THByteTensor *single_res_name = THByteTensor_new();
            //Output atom names and residue names
            for(uint i=0; i<conf.groups.size(); i++){
                for(uint j=0; j<conf.groups[i]->atomNames.size(); j++){
                    uint idx = conf.groups[i]->atomIndexes[j];
                    
                    THByteTensor_select(single_atom_name, atom_names, 0, idx);
                    THByteTensor_select(single_res_name, res_names, 0, idx);
                    
                    StringUtil::string2Tensor(ProtUtil::convertRes1to3(conf.groups[i]->residueName), single_res_name);
                    StringUtil::string2Tensor(conf.groups[i]->atomNames[j], single_atom_name);
                    
                }
            }
            THByteTensor_free(single_atom_name);
            THByteTensor_free(single_res_name);
            THDoubleTensor_free(dummy_grad);
        }else{
            std::cout<<"Not implemented\n";
        }
    }
    void Angles2Coords_backward(    THDoubleTensor *grad_atoms,
                                    THDoubleTensor *grad_angles,
                                    const char* sequence,
                                    THDoubleTensor *input_angles,
                                    int add_terminal
                            ){
        if(input_angles->nDimension == 2){
            std::string aa(sequence);
            uint length = aa.length();
            bool add_term = int2bool(add_terminal);
            int num_atoms = ProtUtil::getNumAtoms(aa, add_term);
            THDoubleTensor *dummy_coords = THDoubleTensor_newWithSize1d( 3*num_atoms);
            cConformation conf( aa, THDoubleTensor_data(input_angles), THDoubleTensor_data(grad_angles), 
                                length, THDoubleTensor_data(dummy_coords));
            conf.backward(conf.root, THDoubleTensor_data(grad_atoms));
            THDoubleTensor_free(dummy_coords);
        }else{
            std::cout<<"Not implemented\n";
        }
    }
    void Angles2Coords_save(    const char* sequence,
                                THDoubleTensor *input_angles, 
                                const char* output_filename,
                                int add_terminal
                            ){
        if(input_angles->nDimension == 2){
            std::string aa(sequence);
            uint length = aa.length();
            bool add_term = int2bool(add_terminal);
            int num_atoms = ProtUtil::getNumAtoms(aa, add_term);
            THDoubleTensor *dummy_grad = THDoubleTensor_newWithSize2d(input_angles->size[0], input_angles->size[1]);
            THDoubleTensor *dummy_coords = THDoubleTensor_newWithSize1d( 3*num_atoms);
            cConformation conf( aa, THDoubleTensor_data(input_angles), THDoubleTensor_data(dummy_grad), 
                                length, THDoubleTensor_data(dummy_coords));
            conf.save(std::string(output_filename));
            THDoubleTensor_free(dummy_grad);
            THDoubleTensor_free(dummy_coords);
        }else{
            std::cout<<"Not implemented\n";
        }
    }
}