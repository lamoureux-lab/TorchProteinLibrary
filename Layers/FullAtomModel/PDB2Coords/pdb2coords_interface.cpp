#include "pdb2coords_interface.h"
#include "cPDBLoader.h"
#include "nUtil.h"
#include <iostream>
#include <string>
#include <algorithm>
/*
void PDB2CoordsOrdered(at::Tensor filenames, at::Tensor coords, at::Tensor res_names, at::Tensor atom_names, at::Tensor num_atoms, at::Tensor mask){
    bool add_terminal = true;
    if( filenames.dtype() != at::kByte || res_names.dtype() != at::kByte || atom_names.dtype() != at::kByte 
        || coords.dtype() != at::kDouble || num_atoms.dtype() != at::kInt || mask.dtype() != at::kByte){
            throw("Incorrect tensor types");
    }
    if(filenames.ndimension() != 2){
        throw("Incorrect input ndim");
    }

    int batch_size = filenames.size(0);

    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_filename = filenames[i];
        std::string filename = StringUtil::tensor2String(single_filename);
        cPDBLoader pdb(filename);
        num_atoms[i] = int(pdb.r.size());
    }
    int max_num_atoms = num_atoms.max().data<int>()[0];
    
    int64_t size_coords[] = {batch_size, max_num_atoms*3};
    int64_t size_names[] = {batch_size, max_num_atoms, 4};
    int64_t size_mask[] = {batch_size, max_num_atoms};
    
    coords.resize_(at::IntList(size_coords, 2));
    res_names.resize_(at::IntList(size_names, 3));
    atom_names.resize_(at::IntList(size_names, 3));
    mask.resize_(at::IntList(size_mask, 3));
    
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_coords = coords[i];
        at::Tensor single_filename = filenames[i];
        at::Tensor single_res_names = res_names[i];
        at::Tensor single_atom_names = atom_names[i];
        at::Tensor single_mask = mask[i];
        
        std::string filename = StringUtil::tensor2String(single_filename);
        
        cPDBLoader pdb(filename);
        
        pdb.reorder();
        int global_ind = 0;
        std::string lastO("O");
        for(int j=0; j<pdb.res_r.size(); j++){
            for(int k=0; k<pdb.res_r[j].size(); k++){
                uint idx = ProtUtil::getAtomIndex(pdb.res_res_names[j], pdb.res_atom_names[j][k]) + global_ind;
                at::Tensor single_atom_name = single_atom_names[idx];
                at::Tensor single_res_name = single_res_names[idx];
                                
                StringUtil::string2Tensor(pdb.res_res_names[j], single_res_name);
                StringUtil::string2Tensor(pdb.res_atom_names[j][k], single_atom_name);
                single_coords[3*idx + 0] = res_r[j][k].v[0];
                single_coords[3*idx + 1] = res_r[j][k].v[1];
                single_coords[3*idx + 2] = res_r[j][k].v[2];
            }
            global_ind += ProtUtil::getAtomIndex(pdb.res_res_names[j], lastO) + 1;
        }
    }
}
*/
void PDB2CoordsUnordered(at::Tensor filenames, at::Tensor coords, at::Tensor res_names, at::Tensor res_nums, at::Tensor atom_names, at::Tensor num_atoms){
    
    // if( filenames.dtype() != at::kByte || res_names.dtype() != at::kByte || atom_names.dtype() != at::kByte 
    //     || coords.dtype() != at::kDouble || num_atoms.dtype() != at::kInt || res_nums.dtype() != at::kInt){
    //         std::cout<<"Incorrect tensor types"<<std::endl;
    //         throw("Incorrect tensor types");
    // }
    if(coords.ndimension() != 2){
        std::cout<<"Incorrect input ndim"<<std::endl;
        throw("Incorrect input ndim");
    }
    
    int batch_size = filenames.size(0);

    // int std::vector<int> num_atoms(batch_size);
    // std::cout<<"Start "<<batch_size<<std::endl;
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_filename = filenames[i];
        std::string filename = StringUtil::tensor2String(single_filename);
        cPDBLoader pdb(filename);
        num_atoms[i] = int(pdb.r.size());
    }
    int max_num_atoms = num_atoms.max().data<int>()[0];
    // std::cout<<max_num_atoms<<std::endl;
    int64_t size_nums[] = {batch_size, max_num_atoms};
    int64_t size_coords[] = {batch_size, max_num_atoms*3};
    int64_t size_names[] = {batch_size, max_num_atoms, 4};
    
    coords.resize_(at::IntList(size_coords, 2));
    res_names.resize_(at::IntList(size_names, 3));
    res_nums.resize_(at::IntList(size_nums, 2));
    atom_names.resize_(at::IntList(size_names, 3));
    
    
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_coords = coords[i];
        at::Tensor single_filename = filenames[i];
        at::Tensor single_res_names = res_names[i];
        at::Tensor single_res_nums = res_nums[i];
        at::Tensor single_atom_names = atom_names[i];
        
        std::string filename = StringUtil::tensor2String(single_filename);
        cPDBLoader pdb(filename);
        for(int j=0; j<pdb.r.size(); j++){
            cVector3 r_target(single_coords.data<double>() + 3*j);
            r_target = pdb.r[j];
            StringUtil::string2Tensor(pdb.res_names[j], single_res_names[j]);
            StringUtil::string2Tensor(pdb.atom_names[j], single_atom_names[j]);
            single_res_nums.data<int>()[j] = pdb.res_nums[j];
        }
    }
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("PDB2Coords", &PDB2Coords, "Convert PDB to coordinates");
// }

