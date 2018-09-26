#include "pdb2coords_interface.h"
#include "cPDBLoader.h"
#include "nUtil.h"
#include <iostream>
#include <string>

void PDB2Coords(at::Tensor filenames, at::Tensor coords, at::Tensor res_names, at::Tensor atom_names, int strict){
    bool add_terminal = false;
    if( filenames.dtype() != at::kByte || res_names.dtype() != at::kByte || atom_names.dtype() != at::kByte 
        || coords.dtype() != at::kDouble){
            throw("Incorrect tensor types");
    }
    if(coords.ndimension() != 2){
        throw("Incorrect input ndim");
    }
    if( (strict != 0) && (strict != 1))
        throw("Variable strict != {1,0}");

    int batch_size = filenames.size(0);
    
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_coords = coords[i];
        at::Tensor single_filename = filenames[i];
        at::Tensor single_res_names = res_names[i];
        at::Tensor single_atom_names = atom_names[i];
        
        std::string filename = StringUtil::tensor2String(single_filename);
        
        cPDBLoader pdb(filename);
        if(strict==1){
            pdb.reorder(single_coords.data<double>());
            int global_ind=0;
            std::string lastO("O");
            for(int j=0; j<pdb.res_r.size(); j++){
                for(int k=0; k<pdb.res_r[j].size(); k++){
                    uint idx = ProtUtil::getAtomIndex(pdb.res_res_names[j], pdb.res_atom_names[j][k]) + global_ind;
                    at::Tensor single_atom_name = single_atom_names[idx];
                    at::Tensor single_res_name = single_res_names[idx];
                    
                    StringUtil::string2Tensor(pdb.res_res_names[j], single_res_name);
                    StringUtil::string2Tensor(pdb.res_atom_names[j][k], single_atom_name);
                }
                if(add_terminal){
                    if( j<(pdb.res_r.size()-1) )
                        lastO = "O";
                    else
                        lastO = "OXT";
                }else{
                    lastO = "O";
                }
                global_ind += ProtUtil::getAtomIndex(pdb.res_res_names[j], lastO) + 1;
            }
        }else if(strict==0){
            for(int i=0; i<pdb.r.size(); i++){
                cVector3 r_target(single_coords.data<double>() + 3*i);
                r_target = pdb.r[i];
                StringUtil::string2Tensor(pdb.res_names[i], single_res_names[i]);
                StringUtil::string2Tensor(pdb.atom_names[i], single_atom_names[i]);
            }
        }
    }
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("PDB2Coords", &PDB2Coords, "Convert PDB to coordinates");
// }

