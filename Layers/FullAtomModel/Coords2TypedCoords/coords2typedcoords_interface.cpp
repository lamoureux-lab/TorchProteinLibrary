#include "coords2typedcoords_interface.h"
#include "cConformation.h"
#include <iostream>
#include <string>
#include "nUtil.h"


void Coords2TypedCoords_forward(    at::Tensor input_coords, 
                                    at::Tensor res_names,
                                    at::Tensor atom_names,
                                    at::Tensor input_num_atoms,
                                    at::Tensor output_coords,
                                    at::Tensor output_num_atoms_of_type,
                                    at::Tensor output_offsets,
                                    at::Tensor output_atom_indexes
                                ){
    const uint num_atom_types = 11;

    // if( res_names.dtype() != at::kByte || atom_names.dtype() != at::kByte 
    //     || input_coords.dtype() != at::kDouble || output_coords.dtype() != at::kDouble
    //     || input_num_atoms.dtype() != at::kInt || output_num_atoms_of_type.dtype() != at::kInt
    //     || output_offsets.dtype() != at::kInt || output_atom_indexes.dtype() != at::kInt){
    //         throw("Incorrect tensor types");
    // }
    if(input_coords.ndimension() != 2){
        throw("Incorrect input ndim");
    }
    int batch_size = input_coords.size(0);
                    
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        int num_atoms = input_num_atoms.accessor<int,1>()[i];

        at::Tensor single_intput_coords = input_coords[i];
        at::Tensor single_atom_names = atom_names[i];
        at::Tensor single_res_names = res_names[i];
        
        at::Tensor single_output_coords = output_coords[i];
        at::Tensor single_output_num_atoms_of_type = output_num_atoms_of_type[i];
        at::Tensor single_output_offsets = output_offsets[i];
        at::Tensor single_output_atom_indexes = output_atom_indexes[i];
        // std::cout<<num_atoms<<std::endl;
        int num_atoms_added[num_atom_types];
        int atom_types[num_atoms];
        for(int j=0;j<num_atom_types;j++){
            num_atoms_added[j] = 0;
        }
        
        //Assign atom types
        at::Tensor single_atom_name, single_res_name;
                
        for(int j=0; j<num_atoms; j++){
            single_atom_name = single_atom_names[j];
            single_res_name = single_res_names[j];
            int type;
            try{
                type = ProtUtil::get11AtomType(StringUtil::tensor2String(single_res_name), 
                                                    StringUtil::tensor2String(single_atom_name), false);
            }catch(std::string e){
                std::cout<<e<<std::endl;
                std::cout<<StringUtil::tensor2String(single_res_name)<<" "<<StringUtil::tensor2String(single_atom_name)<<std::endl;
                throw(std::string("TypeAssignmentError"));
            }
            atom_types[j] = type;
            single_output_num_atoms_of_type[type] += 1;
        }
        // std::cout<<"assigned types"<<std::endl;
        //Compute memory offsets for an atom type
        for(int j=1;j<num_atom_types; j++){
            single_output_offsets[j] = single_output_offsets[j-1] + single_output_num_atoms_of_type[j-1];
        }
        // std::cout<<"assigned offsets"<<std::endl;
        //Copy information
        auto a_single_output_offsets = single_output_offsets.accessor<int,1>();
        for(int j=0;j<num_atoms; j++){
            int type = atom_types[j];
            int offset = a_single_output_offsets[type];
            int dst_idx = offset + num_atoms_added[type];
            cVector3 r_dst( single_output_coords.data<double>() + 3*dst_idx );
            cVector3 r_src( single_intput_coords.data<double>() + 3*j);
            r_dst = r_src;
            single_output_atom_indexes[offset + num_atoms_added[type]] = j;
            num_atoms_added[type] += 1;
        }
        // std::cout<<"rearranged coordinates"<<std::endl;
    }
    
}
void Coords2TypedCoords_backward(   at::Tensor grad_typed_coords,
                                    at::Tensor grad_flat_coords,
                                    at::Tensor num_atoms_of_type,
                                    at::Tensor offsets,
                                    at::Tensor atom_indexes
                        ){
    const uint num_atom_types=11;
    // if( grad_typed_coords.dtype() != at::kDouble || grad_flat_coords.dtype() != at::kDouble
    //     || num_atoms_of_type.dtype() != at::kInt || offsets.dtype() != at::kInt
    //     || atom_indexes.dtype() != at::kInt){
    //         throw("Incorrect tensor types");
    // }
    if(grad_typed_coords.ndimension() != 2){
        throw("Incorrect input ndim");
    }
   
    int batch_size = grad_flat_coords.size(0);
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){

        at::Tensor single_grad_typed_coords = grad_typed_coords[i];
        at::Tensor single_grad_flat_coords = grad_flat_coords[i];
        at::Tensor single_atom_indexes = atom_indexes[i];
        at::Tensor single_offsets = offsets[i];
        at::Tensor single_num_atoms_of_type = num_atoms_of_type[i];
        
        for(int j=0;j<num_atom_types; j++){
            int num_atoms = single_num_atoms_of_type.accessor<int,1>()[j];
            int offset = single_offsets.accessor<int,1>()[j];
            for(int k=0; k<num_atoms; k++){
                int src_idx = offset + k;
                int dst_idx = single_atom_indexes.accessor<int,1>()[src_idx];
                cVector3 r_src( single_grad_typed_coords.data<double>()+ 3*src_idx );
                cVector3 r_dst( single_grad_flat_coords.data<double>() + 3*dst_idx );
                r_dst = r_src;
            }
        }
    }
}