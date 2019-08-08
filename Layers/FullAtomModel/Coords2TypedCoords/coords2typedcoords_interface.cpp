#include "coords2typedcoords_interface.h"
#include "cConformation.h"
#include <iostream>
#include <string>
#include "nUtil.h"


void Coords2TypedCoords_forward(    torch::Tensor input_coords, 
                                    torch::Tensor res_names,
                                    torch::Tensor atom_names,
                                    torch::Tensor input_num_atoms,
                                    torch::Tensor output_coords,
                                    torch::Tensor output_num_atoms_of_type,
                                    torch::Tensor output_offsets,
                                    torch::Tensor output_atom_indexes
                                ){
    const uint num_atom_types = 11;
    CHECK_CPU_INPUT(input_coords);
    CHECK_CPU_INPUT_TYPE(res_names, torch::kByte);
    CHECK_CPU_INPUT_TYPE(atom_names, torch::kByte);
    CHECK_CPU_INPUT_TYPE(input_num_atoms, torch::kInt);
    CHECK_CPU_INPUT(output_coords);
    CHECK_CPU_INPUT_TYPE(output_num_atoms_of_type, torch::kInt);
    CHECK_CPU_INPUT_TYPE(output_offsets, torch::kInt);
    CHECK_CPU_INPUT_TYPE(output_atom_indexes, torch::kInt);
    
    if(input_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    int batch_size = input_coords.size(0);
                    
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        int num_atoms = input_num_atoms.accessor<int,1>()[i];

        torch::Tensor single_intput_coords = input_coords[i];
        torch::Tensor single_atom_names = atom_names[i];
        torch::Tensor single_res_names = res_names[i];
        
        torch::Tensor single_output_coords = output_coords[i];
        torch::Tensor single_output_num_atoms_of_type = output_num_atoms_of_type[i];
        torch::Tensor single_output_offsets = output_offsets[i];
        torch::Tensor single_output_atom_indexes = output_atom_indexes[i];
        
        int num_atoms_added[num_atom_types];
        int atom_types[num_atoms];
        for(int j=0;j<num_atom_types;j++){
            num_atoms_added[j] = 0;
        }
        
        //Assign atom types
        torch::Tensor single_atom_name, single_res_name;
                
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
        
        //Compute memory offsets for an atom type
        for(int j=1;j<num_atom_types; j++){
            single_output_offsets[j] = single_output_offsets[j-1] + single_output_num_atoms_of_type[j-1];
        }
        
        //Copy information
        auto a_single_output_offsets = single_output_offsets.accessor<int,1>();
        for(int j=0;j<num_atoms; j++){
            int type = atom_types[j];
            int offset = a_single_output_offsets[type];
            int dst_idx = offset + num_atoms_added[type];
            AT_DISPATCH_FLOATING_TYPES(input_coords.type(), "Coords2TypedCoords_forward", ([&]{
                cVector3<scalar_t> r_dst( single_output_coords.data<scalar_t>() + 3*dst_idx );
                cVector3<scalar_t> r_src( single_intput_coords.data<scalar_t>() + 3*j);
                r_dst = r_src;
            }));
            single_output_atom_indexes[offset + num_atoms_added[type]] = j;
            num_atoms_added[type] += 1;
        }
        
    }
    
}
void Coords2TypedCoords_backward(   torch::Tensor grad_typed_coords,
                                    torch::Tensor grad_flat_coords,
                                    torch::Tensor num_atoms_of_type,
                                    torch::Tensor offsets,
                                    torch::Tensor atom_indexes
                        ){
    const uint num_atom_types=11;
    CHECK_CPU_INPUT(grad_typed_coords);
    CHECK_CPU_INPUT(grad_flat_coords);
    
    CHECK_CPU_INPUT_TYPE(num_atoms_of_type, torch::kInt);
    CHECK_CPU_INPUT_TYPE(offsets, torch::kInt);
    CHECK_CPU_INPUT_TYPE(atom_indexes, torch::kInt);
    if(grad_typed_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
   
    int batch_size = grad_flat_coords.size(0);
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){

        torch::Tensor single_grad_typed_coords = grad_typed_coords[i];
        torch::Tensor single_grad_flat_coords = grad_flat_coords[i];
        torch::Tensor single_atom_indexes = atom_indexes[i];
        torch::Tensor single_offsets = offsets[i];
        torch::Tensor single_num_atoms_of_type = num_atoms_of_type[i];
        
        for(int j=0;j<num_atom_types; j++){
            int num_atoms = single_num_atoms_of_type.accessor<int,1>()[j];
            int offset = single_offsets.accessor<int,1>()[j];
            for(int k=0; k<num_atoms; k++){
                int src_idx = offset + k;
                int dst_idx = single_atom_indexes.accessor<int,1>()[src_idx];
                AT_DISPATCH_FLOATING_TYPES(grad_typed_coords.type(), "Coords2TypedCoords_backward", ([&]{
                    cVector3<scalar_t> r_src( single_grad_typed_coords.data<scalar_t>()+ 3*src_idx );
                    cVector3<scalar_t> r_dst( single_grad_flat_coords.data<scalar_t>() + 3*dst_idx );
                    r_dst = r_src;
                }));
            }
        }
    }
}