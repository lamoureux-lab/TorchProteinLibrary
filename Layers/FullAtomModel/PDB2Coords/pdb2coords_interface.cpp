#include <TH/TH.h>
#include "cPDBLoader.h"
#include "nUtil.h"
#include <iostream>
#include <string>

bool int2bool(int add_terminal){
    bool add_term;
    if(add_terminal == 1){
        add_term = true;
    }else if(add_terminal == 0){
        add_term = false;
    }else{
        std::cout<<"unknown add_terminal = "<<add_terminal<<std::endl;
        throw std::string("unknown add_terminal");
    }
    return add_term;
}
extern "C" {
    void PDB2Coords( THByteTensor *filenames, THDoubleTensor *coords, THByteTensor *res_names,
                                THByteTensor *atom_names, int add_terminal){
        bool add_term = int2bool(add_terminal);
        if(coords->nDimension == 2){
            int batch_size = filenames->size[0];
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<batch_size; i++){
                THDoubleTensor *single_coords = THDoubleTensor_new();
                THByteTensor *single_filename = THByteTensor_new();
                THByteTensor *single_res_names = THByteTensor_new();
                THByteTensor *single_atom_names = THByteTensor_new();
                THDoubleTensor_select(single_coords, coords, 0, i);
                THByteTensor_select(single_filename, filenames, 0, i);
                THByteTensor_select(single_res_names, res_names, 0, i);
                THByteTensor_select(single_atom_names, atom_names, 0, i);
                std::string filename((const char*)THByteTensor_data(single_filename));
                
                cPDBLoader pdb(filename);
                pdb.reorder(THDoubleTensor_data(single_coords));
                int global_ind=0;
                std::string lastO("O");
                THByteTensor *single_res_name = THByteTensor_new();
                THByteTensor *single_atom_name = THByteTensor_new();
                for(int j=0; j<pdb.res_r.size(); j++){
                    for(int k=0; k<pdb.res_r[j].size(); k++){
                        uint idx = ProtUtil::getAtomIndex(pdb.res_res_names[j], pdb.res_atom_names[j][k]) + global_ind;

                        THByteTensor_select(single_atom_name, single_atom_names, 0, idx);
                        THByteTensor_select(single_res_name, single_res_names, 0, idx);

                        StringUtil::string2Tensor(pdb.res_res_names[j], single_res_name);
                        StringUtil::string2Tensor(pdb.res_atom_names[j][k], single_atom_name);
                    }
                    if(add_term){
                        if( j<(pdb.res_r.size()-1) )
                            lastO = "O";
                        else
                            lastO = "OXT";
                    }else{
                        lastO = "O";
                    }
                    global_ind += ProtUtil::getAtomIndex(pdb.res_res_names[j], lastO) + 1;
                }

                THByteTensor_free(single_filename);
                THByteTensor_free(single_res_names);
                THByteTensor_free(single_atom_names);
                THByteTensor_free(single_atom_name);
                THByteTensor_free(single_res_name);
                THDoubleTensor_free(single_coords);
            }
        }else{
            std::cout<<"Not implemented\n";
        }
    }
    int getSeqNumAtoms( const char *sequence, int add_terminal){
        bool add_term;
        if(add_terminal == 1){
            add_term = true;
        }else if(add_terminal == 0){
            add_term = false;
        }else{
            std::cout<<"unknown add_terminal = "<<add_terminal<<std::endl;
            throw std::string("unknown add_terminal");
        }
        cPDBLoader pdb;
        std::string seq(sequence);
        int num_atoms = ProtUtil::getNumAtoms(seq, add_term);
        return num_atoms;
    }
    
}