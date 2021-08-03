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
                                    torch::Tensor output_atom_indexes,
				    int num_atom_types
                                ){
    //const uint num_atom_types = 4; 
    CHECK_CPU_INPUT(input_coords);
    CHECK_CPU_INPUT_TYPE(res_names, torch::kByte);
    CHECK_CPU_INPUT_TYPE(atom_names, torch::kByte);
    CHECK_CPU_INPUT_TYPE(input_num_atoms, torch::kInt);
    CHECK_CPU_INPUT(output_coords);
    CHECK_CPU_INPUT_TYPE(output_num_atoms_of_type, torch::kInt);
    CHECK_CPU_INPUT_TYPE(output_atom_indexes, torch::kInt);
    
    if(input_coords.ndimension() != 3){
        ERROR("Incorrect input ndim");
    }
    int batch_size = input_coords.size(0);
                    
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        int num_atoms = input_num_atoms.accessor<int,1>()[i];

        torch::Tensor single_input_coords = input_coords[i];
        torch::Tensor single_atom_names = atom_names[i];
        torch::Tensor single_res_names = res_names[i];
        
        torch::Tensor single_output_coords = output_coords[i];
        torch::Tensor single_output_num_atoms_of_type = output_num_atoms_of_type[i];
        torch::Tensor single_output_atom_indexes = output_atom_indexes[i];
        
        auto a_atom_indexes = single_output_atom_indexes.accessor<int, 2>();
        auto a_num_atoms_of_type = single_output_num_atoms_of_type.accessor<int, 1>();
        
        int type;      
        for(int j=0; j<num_atoms; j++){
            try{
	      if(num_atom_types == 4){
                type = ProtUtil::get4AtomTypeElement(StringUtil::tensor2String(single_res_names[j]), 
						     StringUtil::tensor2String(single_atom_names[j]), false);
	      }else if(num_atom_types == 29){
		type = ProtUtil::get29AtomTypeCharmm(StringUtil::tensor2String(single_res_names[j]), 
						     StringUtil::tensor2String(single_atom_names[j]), false);
	      }else{
		type = ProtUtil::get11AtomType(StringUtil::tensor2String(single_res_names[j]), 
					       StringUtil::tensor2String(single_atom_names[j]), false);
	      }
		
            }catch(std::string e){
                std::cout<<e<<std::endl;
                std::cout<<StringUtil::tensor2String(single_res_names[j])<<" "<<StringUtil::tensor2String(single_atom_names[j])<<std::endl;
                ERROR("TypeAssignmentError");
            }
            int dst_idx = a_num_atoms_of_type[type];
            AT_DISPATCH_FLOATING_TYPES(input_coords.type(), "Coords2TypedCoords_forward", ([&]{
                cVector3<scalar_t> r_dst( single_output_coords[type].data<scalar_t>() + 3*dst_idx );
                cVector3<scalar_t> r_src( single_input_coords.data<scalar_t>() + 3*j);
                r_dst = r_src;
            }));
            a_atom_indexes[type][dst_idx] = j;
            a_num_atoms_of_type[type] += 1;
        }
        
    }
    
}
void Coords2TypedCoords_backward(   torch::Tensor grad_typed_coords,
                                    torch::Tensor grad_flat_coords,
                                    torch::Tensor num_atoms_of_type,
                                    torch::Tensor atom_indexes,
				    int num_atom_types
                        ){
    //const uint num_atom_types=4;
    CHECK_CPU_INPUT(grad_typed_coords);
    CHECK_CPU_INPUT(grad_flat_coords);
    
    CHECK_CPU_INPUT_TYPE(num_atoms_of_type, torch::kInt);
    CHECK_CPU_INPUT_TYPE(atom_indexes, torch::kInt);
    if(grad_typed_coords.ndimension() != 3){
        ERROR("Incorrect input ndim");
    }
   
    int batch_size = grad_flat_coords.size(0);
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){

        torch::Tensor single_grad_typed_coords = grad_typed_coords[i];
        torch::Tensor single_grad_flat_coords = grad_flat_coords[i];
        torch::Tensor single_atom_indexes = atom_indexes[i];
        torch::Tensor single_num_atoms_of_type = num_atoms_of_type[i];
        
        auto a_single_atom_indexes = single_atom_indexes.accessor<int,2>();
        auto a_single_num_atoms_of_type = single_num_atoms_of_type.accessor<int,1>();
        
        for(int type=0; type<num_atom_types; type++){
            int num_atoms = a_single_num_atoms_of_type[type];
            for(int at_idx=0; at_idx<num_atoms; at_idx++){
                AT_DISPATCH_FLOATING_TYPES(grad_typed_coords.type(), "Coords2TypedCoords_backward", ([&]{
                    cVector3<scalar_t> r_src( single_grad_typed_coords[type].data<scalar_t>()+ 3*at_idx );
                    cVector3<scalar_t> r_dst( single_grad_flat_coords.data<scalar_t>() + 3*a_single_atom_indexes[type][at_idx] );
                    r_dst = r_src;
                }));
            }
        }
    }
}
