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
    void Angles2Coords_forward(  THByteTensor *sequences,
                                THDoubleTensor *input_angles, 
                                THDoubleTensor *output_coords,
                                THByteTensor *res_names,
                                THByteTensor *atom_names,
                                int add_terminal
                            ){
        bool add_term = int2bool(add_terminal);
        if(input_angles->nDimension == 3){
            int batch_size = input_angles->size[0];
                        
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THByteTensor *single_sequence = THByteTensor_new();
                THByteTensor *single_atom_names = THByteTensor_new();
                THByteTensor *single_res_names = THByteTensor_new();
                THDoubleTensor *single_angles = THDoubleTensor_new();
                THDoubleTensor *single_coords = THDoubleTensor_new();
                THByteTensor *single_atom_name = THByteTensor_new();
                THByteTensor *single_res_name = THByteTensor_new();
                
                THByteTensor_select(single_sequence, sequences, 0, i);
                THByteTensor_select(single_atom_names, atom_names, 0, i);
                THByteTensor_select(single_res_names, res_names, 0, i);
                THDoubleTensor_select(single_angles, input_angles, 0, i);
                THDoubleTensor_select(single_coords, output_coords, 0, i);
            
                std::string seq((const char*)THByteTensor_data(single_sequence));
                
                uint length = single_angles->size[1];
                int num_atoms = ProtUtil::getNumAtoms(seq, add_term);
                
                if( single_coords->size[0]<3*num_atoms){
                    throw("incorrect coordinates tensor length");
                }
                
                if( length<seq.length() || single_angles->size[0]<7 ){
                    throw("incorrect angles tensor length");
                }
                
                if( single_res_names->size[0]<seq.length() ){
                    throw("incorrect res names tensor length");
                }
                
                if( single_atom_names->size[0]<seq.length() ){
                    throw("incorrect atom names tensor length");
                }
                
                THDoubleTensor *dummy_grad = THDoubleTensor_newWithSize2d(input_angles->size[1], input_angles->size[2]);
                cConformation conf( seq, THDoubleTensor_data(single_angles), THDoubleTensor_data(dummy_grad),
                                    length, THDoubleTensor_data(single_coords));
                
                //Output atom names and residue names
                for(uint j=0; j<conf.groups.size(); j++){
                    for(uint k=0; k<conf.groups[j]->atomNames.size(); k++){
                        uint idx = conf.groups[j]->atomIndexes[k];       
            
                        THByteTensor_select(single_atom_name, single_atom_names, 0, idx);
                        THByteTensor_select(single_res_name, single_res_names, 0, idx);
                        StringUtil::string2Tensor(ProtUtil::convertRes1to3(conf.groups[j]->residueName), single_res_name);
                        StringUtil::string2Tensor(conf.groups[j]->atomNames[k], single_atom_name);
                        
                    }
                }
                
                THByteTensor_free(single_atom_name);
                THByteTensor_free(single_res_name);
                THDoubleTensor_free(dummy_grad);
            
                THByteTensor_free(single_atom_names);
                THByteTensor_free(single_res_names);
                THByteTensor_free(single_sequence);
                THDoubleTensor_free(single_angles);
                THDoubleTensor_free(single_coords);
        
            }
        }else{
            std::cout<<"Not implemented\n";
        }
    }
    void Angles2Coords_backward(    THDoubleTensor *grad_atoms,
                                    THDoubleTensor *grad_angles,
                                    THByteTensor *sequences,
                                    THDoubleTensor *input_angles,
                                    int add_terminal
                            ){
        bool add_term = int2bool(add_terminal);
        if(input_angles->nDimension == 3){
            int batch_size = input_angles->size[0];

            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                
                THByteTensor *single_sequence = THByteTensor_new();
                THDoubleTensor *single_angles = THDoubleTensor_new();
                THDoubleTensor *single_grad_angles = THDoubleTensor_new();
                THDoubleTensor *single_grad_coords = THDoubleTensor_new();

                THByteTensor_select(single_sequence, sequences, 0, i);
                THDoubleTensor_select(single_angles, input_angles, 0, i);
                THDoubleTensor_select(single_grad_angles, grad_angles, 0, i);
                THDoubleTensor_select(single_grad_coords, grad_atoms, 0, i);
                std::string seq((const char*)THByteTensor_data(single_sequence));
                
                uint length = single_angles->size[1];
                int num_atoms = ProtUtil::getNumAtoms(seq, add_term);
                
                THDoubleTensor *dummy_coords = THDoubleTensor_newWithSize1d( 3*num_atoms);
                cConformation conf( seq, THDoubleTensor_data(single_angles), THDoubleTensor_data(single_grad_angles),
                                    length, THDoubleTensor_data(dummy_coords));
                conf.backward(conf.root, THDoubleTensor_data(single_grad_coords));
                
                THByteTensor_free(single_sequence);
                THDoubleTensor_free(dummy_coords);
                THDoubleTensor_free(single_angles);
                THDoubleTensor_free(single_grad_angles);
                THDoubleTensor_free(single_grad_coords);
                
            }
        
        }else{
            std::cout<<"Not implemented\n";
        }
    }
    void Angles2Coords_save(    const char* sequence,
                                THDoubleTensor *input_angles, 
                                const char* output_filename,
                                int add_terminal,
                                const char mode
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
            conf.save(std::string(output_filename), mode);
            THDoubleTensor_free(dummy_grad);
            THDoubleTensor_free(dummy_coords);
        }else{
            std::cout<<"Not implemented\n";
        }
    }
}