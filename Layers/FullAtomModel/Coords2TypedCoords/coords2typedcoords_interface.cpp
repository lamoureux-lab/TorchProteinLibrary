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
        if(input_coords->nDimension == 1){
            uint num_atom_types = 11;
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
        if(grad_flat_coords->nDimension == 1){
            uint num_atom_types=11;
            for(uint i=0;i<num_atom_types; i++){
                uint num_atoms = THIntTensor_get1d(num_atoms_of_type, i);
                uint offset = THIntTensor_get1d(offsets, i);
                for(int j=0; j<num_atoms; j++){
                    cVector3 r_src(THDoubleTensor_data(grad_typed_coords) + 3*(offset + j));
                    cVector3 r_dst(THDoubleTensor_data(grad_flat_coords) + 3*THIntTensor_get1d(atom_indexes, offset+j));
                    r_dst = r_src;
                }
            }
        }else{
            std::cout<<"Not implemented\n";
        }
    }
    
}