#include "pdb2coords_interface.h"
#include "cPDBLoader.h"
#include "nUtil.h"
#include <iostream>
#include <string>
#include <algorithm>

void PDB2CoordsOrdered( torch::Tensor filenames, torch::Tensor coords, torch::Tensor chain_names, torch::Tensor res_names, 
                        torch::Tensor res_nums, torch::Tensor atom_names, torch::Tensor atom_mask, torch::Tensor num_atoms, int polymer_type){
    
    CHECK_CPU_INPUT_TYPE(filenames, torch::kByte);
    CHECK_CPU_INPUT_TYPE(res_names, torch::kByte);
    CHECK_CPU_INPUT_TYPE(atom_names, torch::kByte);
    CHECK_CPU_INPUT_TYPE(chain_names, torch::kByte);
    CHECK_CPU_INPUT_TYPE(res_nums, torch::kInt);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    CHECK_CPU_INPUT_TYPE(atom_mask, torch::kByte);
    CHECK_CPU_INPUT(coords);
    if(coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    if (polymer_type == 0){
        int batch_size = filenames.size(0);
        std::string resLastAtom("OXT");

        #pragma omp parallel for
        for(int i=0; i<batch_size; i++){
            torch::Tensor single_filename = filenames[i];
            std::string filename = StringUtil::tensor2String(single_filename);

            cPDBLoader pdb(filename, 0);
            num_atoms[i] = 0;
            int previous_res_num = pdb.res_nums[0];
            for(int j=0; j<pdb.r.size(); j++){
                if (previous_res_num < pdb.res_nums[j]) {
                    previous_res_num = pdb.res_nums[j];
                    num_atoms[i] += int(ProtUtil::getAtomIndex(pdb.res_names[j-1], resLastAtom));
                }
            }
            num_atoms[i] += int(ProtUtil::getAtomIndex(pdb.res_names[pdb.r.size()-1], resLastAtom));

       // Num_atoms Test 1
        std::cout << "\n atoms names:" << pdb.atom_names << "\n res_names:" << pdb.res_names;
        std::cout << "\n size atom names:" << pdb.atom_names.size();
//        std::cout << "\n res_nums:" << pdb.res_nums;
        }

        // Num_atoms Test 2
        std::cout << "\n num_atoms:" << num_atoms << "\n";

        int max_num_atoms = num_atoms.max().data<int>()[0];
        int64_t size_nums[] = {batch_size, max_num_atoms};
        int64_t size_coords[] = {batch_size, max_num_atoms*3};
        int64_t size_names[] = {batch_size, max_num_atoms, 4};

        coords.resize_(torch::IntList(size_coords, 2)).fill_(0.0);
        chain_names.resize_(torch::IntList(size_names, 3)).fill_(0);
        res_names.resize_(torch::IntList(size_names, 3)).fill_(0);
        res_nums.resize_(torch::IntList(size_nums, 2)).fill_(0);
        atom_names.resize_(torch::IntList(size_names, 3)).fill_(0);
        atom_mask.resize_(torch::IntList(size_nums, 2)).fill_(0);


        #pragma omp parallel for
        for(int i=0; i<batch_size; i++){
            torch::Tensor single_coords = coords[i];
            torch::Tensor single_filename = filenames[i];
            torch::Tensor single_chain_names = chain_names[i];
            torch::Tensor single_res_names = res_names[i];
            torch::Tensor single_res_nums = res_nums[i];
            torch::Tensor single_atom_names = atom_names[i];
            torch::Tensor single_mask = atom_mask[i];

            std::string filename = StringUtil::tensor2String(single_filename);

            cPDBLoader pdb(filename, 0);

            int global_ind = 0;
            int previous_res_num = pdb.res_nums[0];
            for(int j=0; j<pdb.r.size(); j++){
                if (previous_res_num < pdb.res_nums[j]) {
                    previous_res_num = pdb.res_nums[j];
                    global_ind += ProtUtil::getAtomIndex(pdb.res_names[j-1], resLastAtom);
                }
                uint idx = ProtUtil::getAtomIndex(pdb.res_names[j], pdb.atom_names[j]) + global_ind;

//                Why resLastAtom
                std::cout << "\n res num[j]:" << pdb.res_nums[j];
                std::cout << "\n global_ind:" << global_ind;
                std::cout << "\n idx:" << idx;

                StringUtil::string2Tensor(pdb.chain_names[j], single_chain_names[idx]);
                StringUtil::string2Tensor(pdb.res_names[j], single_res_names[idx]);
                StringUtil::string2Tensor(pdb.atom_names[j], single_atom_names[idx]);
                single_res_nums[idx] = pdb.res_nums[j];

                single_coords[3*idx + 0] = pdb.r[j].v[0];
                single_coords[3*idx + 1] = pdb.r[j].v[1];
                single_coords[3*idx + 2] = pdb.r[j].v[2];
                single_mask[idx] = 1;
            }
        }
    }

    else if (polymer_type == 1){
        int batch_size = filenames.size(0);
//
        #pragma omp parallel for
        for(int i=0; i<batch_size; i++){
            torch::Tensor single_filename = filenames[i];
            std::string filename = StringUtil::tensor2String(single_filename);

            cPDBLoader pdb(filename, 1);
//            std::cout << pdb.res_names << " \n cPDBLoader Test in pdb2coords \n"; //Test in pdb2coords of cPDBLoader 1
//            std::cout << res_names << "cPDBLoader Test in pdb2coords \n"; //Test in pdb2coords of cPDBLoader 2
//            std::cout << "cPDBLoader Test in pdb2coords \n"; //Test in pdb2coords of cPDBLoader 3
            num_atoms[i] = 0;

            // Num_atoms and res_num Test
//            std::cout << "num atoms:" << num_atoms << "\n res_nums: " << pdb.res_nums << "\n";
//            std::cout << "size res_nums:" << pdb.res_nums.size() << "size atom names" << pdb.atom_names.size();

            num_atoms[i] = static_cast<int>(pdb.atom_names.size());

//          Get Num_atoms another way

//            int previous_res_num = pdb.res_nums[0];
//            for(int j=0; j<pdb.r.size(); j++){
//                if (previous_res_num < pdb.res_nums[j]) {
//                    previous_res_num = pdb.res_nums[j];
//                    num_atoms[i] += int(ProtUtil::getAtomIndex(pdb.res_names[j-1], resLastAtom));
//                }
//          }
//            num_atoms[i] += int(ProtUtil::getAtomIndex(pdb.res_names[pdb.r.size()-1], resLastAtom));
        }
        //Num_atoms Test
//        std::cout << num_atoms << "\n";
//
        int max_num_atoms = num_atoms.max().data<int>()[0];
//        int max_num_atoms = num_atoms
        int64_t size_nums[] = {batch_size, max_num_atoms};
        int64_t size_coords[] = {batch_size, max_num_atoms*3};
        int64_t size_names[] = {batch_size, max_num_atoms, 4};
//
        coords.resize_(torch::IntList(size_coords, 2)).fill_(0.0);
        chain_names.resize_(torch::IntList(size_names, 3)).fill_(0);
        res_names.resize_(torch::IntList(size_names, 3)).fill_(0);
        res_nums.resize_(torch::IntList(size_nums, 2)).fill_(0);
        atom_names.resize_(torch::IntList(size_names, 3)).fill_(0);
        atom_mask.resize_(torch::IntList(size_nums, 2)).fill_(0);
//
//
        #pragma omp parallel for
        for(int i=0; i<batch_size; i++){
            torch::Tensor single_coords = coords[i];
            torch::Tensor single_filename = filenames[i];
            torch::Tensor single_chain_names = chain_names[i];
            torch::Tensor single_res_names = res_names[i];
            torch::Tensor single_res_nums = res_nums[i];
            torch::Tensor single_atom_names = atom_names[i];
            torch::Tensor single_mask = atom_mask[i];

            std::string filename = StringUtil::tensor2String(single_filename);

            cPDBLoader pdb(filename, 1);

            int global_ind = 0;
            int previous_res_num = pdb.res_nums[0];
            for(int j=0; j<pdb.r.size(); j++){
                if (previous_res_num < pdb.res_nums[j]) {
                    previous_res_num = pdb.res_nums[j];
                    if (pdb.res_names[j] == "DA" || pdb.res_names[j] == "DG") {
                        std::string resLastAtom("C4");
                        global_ind += ProtUtil::getAtomIndex(pdb.res_names[j-1], resLastAtom);
                    }
                    if (pdb.res_names[j] == "DT" || pdb.res_names[j] == "DC") {
                        std::string resLastAtom("C6");
                        global_ind += ProtUtil::getAtomIndex(pdb.res_names[j-1], resLastAtom);
                    }
                }
                uint idx = ProtUtil::getAtomIndex(pdb.res_names[j], pdb.atom_names[j]) + global_ind;

                StringUtil::string2Tensor(pdb.chain_names[j], single_chain_names[idx]);
                StringUtil::string2Tensor(pdb.res_names[j], single_res_names[idx]);
                StringUtil::string2Tensor(pdb.atom_names[j], single_atom_names[idx]);
                single_res_nums[idx] = pdb.res_nums[j];

                single_coords[3*idx + 0] = pdb.r[j].v[0];
                single_coords[3*idx + 1] = pdb.r[j].v[1];
                single_coords[3*idx + 2] = pdb.r[j].v[2];
                single_mask[idx] = 1;
//            }
        }
    }
//        std::cerr << "Error Polymer Type 1 Not Implemented in pdb2coords_interface.cpp/PDB2CoordsOrdered \n";
        std::cout << atom_names;
    }
//    }

    else if (polymer_type == 2){
        std::cerr << "Error Polymer Type 2 Not Implemented in pdb2coords_interface.cpp/PDB2CoordsOrdered \n";
    }

    else{
    std::cerr << "Error Polymer Type Is Not Valid \n";
    }
}

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
        cPDBLoader pdb(filename, 0);
        num_atoms[i] = int(pdb.r.size());
    }
    int max_num_atoms = num_atoms.max().data<int>()[0];
    int64_t size_nums[] = {batch_size, max_num_atoms};
    int64_t size_coords[] = {batch_size, max_num_atoms*3};
    int64_t size_names[] = {batch_size, max_num_atoms, 4};
    
    coords.resize_(torch::IntList(size_coords, 2)).fill_(0);
    chain_names.resize_(torch::IntList(size_names, 3)).fill_(0);
    res_names.resize_(torch::IntList(size_names, 3)).fill_(0);
    res_nums.resize_(torch::IntList(size_nums, 2)).fill_(0);
    atom_names.resize_(torch::IntList(size_names, 3)).fill_(0);
        
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_coords = coords[i];
        torch::Tensor single_filename = filenames[i];
        torch::Tensor single_chain_names = chain_names[i];
        torch::Tensor single_res_names = res_names[i];
        torch::Tensor single_res_nums = res_nums[i];
        torch::Tensor single_atom_names = atom_names[i];
        
        std::string filename = StringUtil::tensor2String(single_filename);
        cPDBLoader pdb(filename, 0);
        for(int j=0; j<pdb.r.size(); j++){
            AT_DISPATCH_FLOATING_TYPES(coords.type(), "PDB2CoordsUnordered", ([&]{
                cVector3<scalar_t> r_target(single_coords.data<scalar_t>() + 3*j);
                r_target.v[0] = pdb.r[j].v[0];
                r_target.v[1] = pdb.r[j].v[1];
                r_target.v[2] = pdb.r[j].v[2];
            }));
            StringUtil::string2Tensor(pdb.chain_names[j], single_chain_names[j]);
            StringUtil::string2Tensor(pdb.res_names[j], single_res_names[j]);
            StringUtil::string2Tensor(pdb.atom_names[j], single_atom_names[j]);
            single_res_nums.data<int>()[j] = pdb.res_nums[j];
            
        }
    }
}
