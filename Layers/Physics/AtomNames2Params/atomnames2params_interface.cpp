#include <torch/extension.h>
#include "atomnames2params_interface.h"
#include "nUtil.h"
#include <iostream>
#include <math.h>


std::map<int, assignment_indexes> AtomNames2Params_forward(   torch::Tensor resnames, torch::Tensor atomnames, torch::Tensor num_atoms, 
                                                                    torch::Tensor types, torch::Tensor params, torch::Tensor assigned_params){
    CHECK_CPU_INPUT_TYPE(resnames, torch::kByte);
    CHECK_CPU_INPUT_TYPE(atomnames, torch::kByte);
    CHECK_CPU_INPUT_TYPE(types, torch::kByte);
    CHECK_CPU_INPUT_TYPE(params, torch::kDouble);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    
    int num_types = types.size(0);
    std::map<atom_type, indexes_atom_param> type_dict;
    std::map<int, assignment_indexes> assign_dict;
    
    auto param_acc = params.accessor<double, 2>();
    for(int i=0; i<num_types; i++){
        std::string res_name = StringUtil::tensor2String(types[i][0]);
        std::string atom_name = StringUtil::tensor2String(types[i][1]);
        atom_type type = std::make_pair(res_name, atom_name);
        atom_param param(param_acc[i][0], param_acc[i][1]);
        indexes_atom_param iparam(i, param);
        type_dict[type] = iparam;

        assign_dict[i] = assignment_indexes(0);
    }
    
    int batch_size = resnames.size(0);
    auto num_atoms_acc = num_atoms.accessor<int, 1>();
    
    
    
    #pragma omp parallel for shared(type_dict, assign_dict)
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_resnames = resnames[i];
        torch::Tensor single_atomnames = atomnames[i];
        torch::Tensor single_assigned_params = assigned_params[i];

        for(int j=0; j<num_atoms_acc[i]; j++){
            std::string res_name = StringUtil::tensor2String(single_resnames[j]);
            std::string atom_name = StringUtil::tensor2String(single_atomnames[j]);
            atom_type type = std::make_pair(res_name, atom_name);
            int param_index = type_dict[type].first;
            double charge = type_dict[type].second.first;
            double radius = type_dict[type].second.second;
            single_assigned_params[j][0] = charge;
            single_assigned_params[j][1] = radius;
            
            #pragma omp critical
            assign_dict[param_index].push_back(std::make_pair(i,j));

        }
    }
    return assign_dict;
}

void AtomNames2Params_backward(torch::Tensor gradOutput, torch::Tensor gradInput, std::map<int, assignment_indexes> &indexes){
    CHECK_CPU_INPUT_TYPE(gradOutput, torch::kDouble);
    CHECK_CPU_INPUT_TYPE(gradInput, torch::kDouble);
    
    auto gradOutput_acc = gradOutput.accessor<double, 3>();
    
    #pragma omg parallel for
    for( auto element : indexes){
        int param_index = element.first;
        assignment_indexes grad_indexes = element.second;
        double grad_charge = 0.0;
        double grad_radius = 0.0;
        for(int i=0; i<grad_indexes.size(); i++){
            int batch_idx = grad_indexes[i].first;
            int atom_idx = grad_indexes[i].second;
            grad_charge += gradOutput_acc[batch_idx][atom_idx][0];
            grad_radius += gradOutput_acc[batch_idx][atom_idx][1];
        }
        gradInput[param_index][0] = grad_charge;
        gradInput[param_index][1] = grad_radius;
    }   
}