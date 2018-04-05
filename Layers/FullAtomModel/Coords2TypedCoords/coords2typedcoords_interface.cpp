#include <TH/TH.h>
#include "cConformation.h"
#include <iostream>
#include <string>
#include "nUtil.h"

extern "C" {
    void Coords2TypedCoords_forward(    THDoubleTensor *input_coords, 
                                        THByteTensor *res_names,
                                        THByteTensor *atom_names,
                                        THDoubleTensor *output_coords,
                                        THIntTensor *output_num_atoms_of_type,
                                        THIntTensor *output_offsets,
                                        THIntTensor *output_atom_indexes
                                    ){
        uint num_atom_types = 11;
        if(input_coords->nDimension == 1){
            uint num_atoms = input_coords->size[0]/3;
            THByteTensor *single_atom_name = THByteTensor_new();
            THByteTensor *single_res_name = THByteTensor_new();

            uint num_atoms_added[num_atom_types];
            uint atom_types[num_atoms];
            for(int i=0;i<num_atom_types;i++){
                num_atoms_added[i] = 0;
            }
            //Assign atom types
            for(uint i=0; i<num_atoms; i++){
                THByteTensor_select(single_atom_name, atom_names, 0, i);
                THByteTensor_select(single_res_name, res_names, 0, i);
                uint type = ProtUtil::get11AtomType( StringUtil::tensor2String(single_res_name), 
                                                     StringUtil::tensor2String(single_atom_name), false);
                atom_types[i] = type;
                THIntTensor_set1d(output_num_atoms_of_type, type, THIntTensor_get1d(output_num_atoms_of_type, type)+1);
            }
            //Compute memory offsets for an atom type
            for(uint i=1;i<num_atom_types; i++){
                THIntTensor_set1d(output_offsets, i, THIntTensor_get1d(output_offsets, i-1) + THIntTensor_get1d(output_num_atoms_of_type, i-1));
            }
            //Copy information
            for(uint i=0;i<num_atoms; i++){
                uint type = atom_types[i];
                uint offset = THIntTensor_get1d(output_offsets, type);
                cVector3 r_dst(THDoubleTensor_data(output_coords) + 3*(offset + num_atoms_added[type]));
                cVector3 r_src(THDoubleTensor_data(input_coords) + 3*i);
                r_dst = r_src;
                THIntTensor_set1d(output_atom_indexes, offset + num_atoms_added[type], i);
                num_atoms_added[type]+=1;
            }    

            THByteTensor_free(single_atom_name);
            THByteTensor_free(single_res_name);
        }else if(input_coords->nDimension == 2){
            int batch_size = input_coords->size[0];
            uint num_atoms = input_coords->size[1]/3;

            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THDoubleTensor *single_intput_coords = THDoubleTensor_new();
                THDoubleTensor_select(single_intput_coords, input_coords, 0, i);
                THByteTensor *single_atom_names = THByteTensor_new();
                THByteTensor *single_res_names = THByteTensor_new();
                THByteTensor_select(single_atom_names, atom_names, 0, i);
                THByteTensor_select(single_res_names, res_names, 0, i);
                
                THDoubleTensor *single_output_coords = THDoubleTensor_new();
                THIntTensor *single_output_num_atoms_of_type = THIntTensor_new();
                THIntTensor *single_output_offsets = THIntTensor_new();
                THIntTensor *single_output_atom_indexes = THIntTensor_new();

                THDoubleTensor_select(single_output_coords, output_coords, 0, i);
                THIntTensor_select(single_output_offsets, output_offsets, 0, i);
                THIntTensor_select(single_output_num_atoms_of_type, output_num_atoms_of_type, 0, i);
                THIntTensor_select(single_output_atom_indexes, output_atom_indexes, 0, i);
                
                uint num_atoms_added[num_atom_types];
                uint atom_types[num_atoms];
                for(int j=0;j<num_atom_types;j++){
                    num_atoms_added[j] = 0;
                }
                
                //Assign atom types
                THByteTensor *single_atom_name = THByteTensor_new();
                THByteTensor *single_res_name = THByteTensor_new();
                
                for(uint j=0; j<num_atoms; j++){
                    THByteTensor_select(single_atom_name, single_atom_names, 0, j);
                    THByteTensor_select(single_res_name, single_res_names, 0, j);

                    uint type = ProtUtil::get11AtomType( StringUtil::tensor2String(single_res_name), 
                                                        StringUtil::tensor2String(single_atom_name), false);
                    atom_types[j] = type;
                    THIntTensor_set1d(single_output_num_atoms_of_type, type, THIntTensor_get1d(single_output_num_atoms_of_type, type)+1);
                }
                THByteTensor_free(single_atom_name);
                THByteTensor_free(single_res_name);

                //Compute memory offsets for an atom type
                for(uint j=1;j<num_atom_types; j++){
                    THIntTensor_set1d(single_output_offsets, j, THIntTensor_get1d(single_output_offsets, j-1) + THIntTensor_get1d(single_output_num_atoms_of_type, j-1));
                }
                //Copy information
                for(uint j=0;j<num_atoms; j++){
                    uint type = atom_types[j];
                    uint offset = THIntTensor_get1d(single_output_offsets, type);
                    cVector3 r_dst(THDoubleTensor_data(single_output_coords) + 3*(offset + num_atoms_added[type]));
                    cVector3 r_src(THDoubleTensor_data(single_intput_coords) + 3*j);
                    r_dst = r_src;
                    THIntTensor_set1d(single_output_atom_indexes, offset + num_atoms_added[type], j);
                    num_atoms_added[type]+=1;
                }    

                THByteTensor_free(single_atom_names);
                THByteTensor_free(single_res_names);
                THDoubleTensor_free(single_intput_coords);
                THDoubleTensor_free(single_output_coords);
                THIntTensor_free(single_output_offsets);
                THIntTensor_free(single_output_num_atoms_of_type);
                THIntTensor_free(single_output_atom_indexes);
                
            }
        }else{
            std::cout<<"Not implemented\n";
        }
    }
    void Coords2TypedCoords_backward(   THDoubleTensor *grad_typed_coords,
                                        THDoubleTensor *grad_flat_coords,
                                        THIntTensor *num_atoms_of_type,
                                        THIntTensor *offsets,
                                        THIntTensor *atom_indexes
                            ){
        uint num_atom_types=11;
        if(grad_flat_coords->nDimension == 1){

            for(uint i=0;i<num_atom_types; i++){
                uint num_atoms = THIntTensor_get1d(num_atoms_of_type, i);
                uint offset = THIntTensor_get1d(offsets, i);
                for(int j=0; j<num_atoms; j++){
                    cVector3 r_src(THDoubleTensor_data(grad_typed_coords) + 3*(offset + j));
                    cVector3 r_dst(THDoubleTensor_data(grad_flat_coords) + 3*THIntTensor_get1d(atom_indexes, offset+j));
                    r_dst = r_src;
                }
            }
        }else if(grad_flat_coords->nDimension == 2){
            int batch_size = grad_flat_coords->size[0];
            
            // #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){

                THDoubleTensor *single_grad_typed_coords = THDoubleTensor_new();
                THDoubleTensor *single_grad_flat_coords = THDoubleTensor_new();
                THDoubleTensor_select(single_grad_typed_coords, grad_typed_coords, 0, i);
                THDoubleTensor_select(single_grad_flat_coords, grad_flat_coords, 0, i);
                THIntTensor *single_atom_indexes = THIntTensor_new();
                THIntTensor *single_offsets = THIntTensor_new();
                THIntTensor *single_num_atoms_of_type = THIntTensor_new();
                THIntTensor_select(single_atom_indexes, atom_indexes, 0, i);
                THIntTensor_select(single_offsets, offsets, 0, i);
                THIntTensor_select(single_num_atoms_of_type, num_atoms_of_type, 0, i);

                for(uint j=0;j<num_atom_types; j++){
                    uint num_atoms = THIntTensor_get1d(single_num_atoms_of_type, j);
                    uint offset = THIntTensor_get1d(single_offsets, j);
                    for(int k=0; k<num_atoms; k++){
                        cVector3 r_src(THDoubleTensor_data(single_grad_typed_coords) + 3*(offset + k));
                        cVector3 r_dst(THDoubleTensor_data(single_grad_flat_coords) + 3*THIntTensor_get1d(single_atom_indexes, offset+k));
                        r_dst = r_src;
                    }
                }

                THDoubleTensor_free(single_grad_typed_coords);
                THDoubleTensor_free(single_grad_flat_coords);
                THIntTensor_free(single_atom_indexes);
                THIntTensor_free(single_offsets);
                THIntTensor_free(single_num_atoms_of_type);

            }
        }else{
            std::cout<<"Not implemented\n";
        }
    }
    
}