
#include <torch/torch.h>
#include "cConformation.h"
#include <iostream>
#include <string>
#include "nUtil.h"

void Angles2Coords_forward(     at::Tensor sequences,
                                at::Tensor input_angles, 
                                at::Tensor output_coords,
                                at::Tensor res_names,
                                at::Tensor atom_names,
                                bool add_terminal
                        ){
    if( sequences.dtype() != at::kByte || res_names.dtype() != at::kByte || atom_names.dtype() != at::kByte 
        || input_angles.dtype() != at::kDouble || output_coords.dtype() != at::kDouble){
            throw("Incorrect tensor types");
    }
    if(input_angles.ndimension() != 3){
        throw("Incorrect input ndim");
    }
    
    int batch_size = input_angles.sizes()[0];
             
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor batch_index = at::CPU(at::kInt).scalarTensor(i);
        at::Tensor single_sequence = sequences.index_select(0, batch_index);
        at::Tensor single_atom_names = atom_names.index_select(0, batch_index);
        at::Tensor single_res_names = res_names.index_select(0, batch_index);
        at::Tensor single_angles = input_angles.index_select(0, batch_index);
        at::Tensor single_coords = output_coords.index_select(0, batch_index);
        
        std::string seq = StringUtil::tensor2String(single_sequence);
        
        int length = single_angles.sizes()[1];
        int num_atoms = ProtUtil::getNumAtoms(seq, add_terminal);
        
        if( single_coords.sizes()[0]<3*num_atoms){
            throw("incorrect coordinates tensor length");
        }
        
        if( length<seq.length() || single_angles.sizes()[0]<7 ){
            throw("incorrect angles tensor length");
        }
        
        if( single_res_names.sizes()[0]<seq.length() ){
            throw("incorrect res names tensor length");
        }
        
        if( single_atom_names.sizes()[0]<seq.length() ){
            throw("incorrect atom names tensor length");
        }
        
        at::Tensor dummy_grad = at::CPU(at::kDouble).zeros_like(single_angles);
        cConformation conf( seq, single_angles.data<double>(), dummy_grad.data<double>(),
                            length, single_coords.data<double>());
        
        //Output atom names and residue names
        for(int j=0; j<conf.groups.size(); j++){
            for(int k=0; k<conf.groups[j]->atomNames.size(); k++){
                int idx = conf.groups[j]->atomIndexes[k];
                at::Tensor atom_index = at::CPU(at::kInt).scalarTensor(idx);       
                at::Tensor single_atom_name = single_atom_names.index_select(0, atom_index);
                at::Tensor single_res_name = single_res_names.index_select(0, atom_index);
                single_res_name = StringUtil::string2Tensor(ProtUtil::convertRes1to3(conf.groups[j]->residueName));
                single_atom_name = StringUtil::string2Tensor(conf.groups[j]->atomNames[k]);
            }
        }
    }
}

void Angles2Coords_backward(    at::Tensor grad_atoms,
                                at::Tensor grad_angles,
                                at::Tensor sequences,
                                at::Tensor input_angles,
                                bool add_terminal
                        ){
    if( sequences.dtype() != at::kByte || grad_atoms.dtype() != at::kDouble || grad_angles.dtype() != at::kDouble
        || input_angles.dtype() != at::kDouble){
            throw("Incorrect tensor types");
    }
    if(input_angles.ndimension() != 3){
        throw("Incorrect input ndim");
    }
    
    int batch_size = input_angles.sizes()[0];

    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor batch_index = at::CPU(at::kInt).scalarTensor(i);
        at::Tensor single_sequence = sequences.index_select(0, batch_index);
        at::Tensor single_angles = input_angles.index_select(0, batch_index);
        at::Tensor single_grad_angles = grad_angles.index_select(0, batch_index);
        at::Tensor single_grad_atoms = grad_atoms.index_select(0, batch_index);
        
        std::string seq = StringUtil::tensor2String(single_sequence);
        
        uint length = single_angles.sizes()[1];
        int num_atoms = ProtUtil::getNumAtoms(seq, add_terminal);
        
        at::Tensor dummy_coords = at::CPU(at::kDouble).zeros({3*num_atoms});
        cConformation conf( seq, single_angles.data<double>(), single_grad_angles.data<double>(),
                            length, dummy_coords.data<double>());
        conf.backward(conf.root, single_grad_atoms.data<double>());
    }

}

void Angles2Coords_save(    const char* sequence,
                            at::Tensor input_angles, 
                            const char* output_filename,
                            bool add_terminal,
                            const char mode
                        ){
    if(input_angles.ndimension() != 2){
        throw("Incorrect input ndim");
    }
    std::string aa(sequence);
    uint length = aa.length();
    int num_atoms = ProtUtil::getNumAtoms(aa, add_terminal);
    at::Tensor dummy_grad = at::CPU(at::kDouble).zeros_like(input_angles);
    at::Tensor dummy_coords = at::CPU(at::kDouble).zeros({3*num_atoms});
    cConformation conf( aa, input_angles.data<double>(), dummy_grad.data<double>(), 
                        length, dummy_coords.data<double>());
    conf.save(std::string(output_filename), mode);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &Angles2Coords_forward, "Angles2Coords forward");
//   m.def("backward", &Angles2Coords_backward, "Angles2Coords backward");
//   m.def("save", &Angles2Coords_save, "Angles2Coords save");
}
