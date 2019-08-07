#include "pdb2coords_interface.h"
#include "cPDBLoader.h"
#include "nUtil.h"
#include <iostream>
#include <string>
#include <algorithm>
/*
void PDB2CoordsOrdered(torch::Tensor filenames, torch::Tensor coords, torch::Tensor res_names, torch::Tensor atom_names, torch::Tensor num_atoms, torch::Tensor mask){
    bool add_terminal = true;
    if( filenames.dtype() != torch::kByte || res_names.dtype() != torch::kByte || atom_names.dtype() != torch::kByte 
        || coords.dtype() != torch::kDouble || num_atoms.dtype() != torch::kInt || mask.dtype() != torch::kByte){
            throw("Incorrect tensor types");
    }
    if(filenames.ndimension() != 2){
        throw("Incorrect input ndim");
    }

    int batch_size = filenames.size(0);

    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_filename = filenames[i];
        std::string filename = StringUtil::tensor2String(single_filename);
        cPDBLoader pdb(filename);
        num_atoms[i] = int(pdb.r.size());
    }
    int max_num_atoms = num_atoms.max().data<int>()[0];
    
    int64_t size_coords[] = {batch_size, max_num_atoms*3};
    int64_t size_names[] = {batch_size, max_num_atoms, 4};
    int64_t size_mask[] = {batch_size, max_num_atoms};
    
    coords.resize_(torch::IntList(size_coords, 2));
    res_names.resize_(torch::IntList(size_names, 3));
    atom_names.resize_(torch::IntList(size_names, 3));
    mask.resize_(torch::IntList(size_mask, 3));
    
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_coords = coords[i];
        torch::Tensor single_filename = filenames[i];
        torch::Tensor single_res_names = res_names[i];
        torch::Tensor single_atom_names = atom_names[i];
        torch::Tensor single_mask = mask[i];
        
        std::string filename = StringUtil::tensor2String(single_filename);
        
        cPDBLoader pdb(filename);
        
        pdb.reorder();
        int global_ind = 0;
        std::string lastO("O");
        for(int j=0; j<pdb.res_r.size(); j++){
            for(int k=0; k<pdb.res_r[j].size(); k++){
                uint idx = ProtUtil::getAtomIndex(pdb.res_res_names[j], pdb.res_atom_names[j][k]) + global_ind;
                torch::Tensor single_atom_name = single_atom_names[idx];
                torch::Tensor single_res_name = single_res_names[idx];
                                
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
void PDB2CoordsUnordered(   torch::Tensor filenames, torch::Tensor coords, torch::Tensor chain_names, torch::Tensor res_names, 
                            torch::Tensor res_nums, torch::Tensor atom_names, torch::Tensor num_atoms){
    CHECK_CPU_INPUT_TYPE(filenames, torch::kByte);
    CHECK_CPU_INPUT_TYPE(res_names, torch::kByte);
    CHECK_CPU_INPUT_TYPE(atom_names, torch::kByte);
    CHECK_CPU_INPUT_TYPE(chain_names, torch::kByte);
    CHECK_CPU_INPUT_TYPE(res_nums, torch::kInt);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    CHECK_CPU_INPUT(coords);
    if(coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    
    int batch_size = filenames.size(0);

    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_filename = filenames[i];
        std::string filename = StringUtil::tensor2String(single_filename);
        cPDBLoader pdb(filename);
        num_atoms[i] = int(pdb.r.size());
    }
    int max_num_atoms = num_atoms.max().data<int>()[0];
    int64_t size_nums[] = {batch_size, max_num_atoms};
    int64_t size_coords[] = {batch_size, max_num_atoms*3};
    int64_t size_names[] = {batch_size, max_num_atoms, 4};
    
    coords.resize_(torch::IntList(size_coords, 2));
    chain_names.resize_(torch::IntList(size_names, 3));
    res_names.resize_(torch::IntList(size_names, 3));
    res_nums.resize_(torch::IntList(size_nums, 2));
    atom_names.resize_(torch::IntList(size_names, 3));
        
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_coords = coords[i];
        torch::Tensor single_filename = filenames[i];
        torch::Tensor single_chain_names = chain_names[i];
        torch::Tensor single_res_names = res_names[i];
        torch::Tensor single_res_nums = res_nums[i];
        torch::Tensor single_atom_names = atom_names[i];
        
        std::string filename = StringUtil::tensor2String(single_filename);
        cPDBLoader pdb(filename);
        for(int j=0; j<pdb.r.size(); j++){
            cVector3<double> r_target(single_coords.data<double>() + 3*j);
            r_target = pdb.r[j];
            StringUtil::string2Tensor(pdb.chain_names[j], single_chain_names[j]);
            StringUtil::string2Tensor(pdb.res_names[j], single_res_names[j]);
            StringUtil::string2Tensor(pdb.atom_names[j], single_atom_names[j]);
            single_res_nums.data<int>()[j] = pdb.res_nums[j];
        }
    }
}
