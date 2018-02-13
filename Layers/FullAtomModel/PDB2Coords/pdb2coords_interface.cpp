#include <TH/TH.h>
#include "cPDBLoader.h"
#include "nUtil.h"
#include <iostream>
#include <string>


extern "C" {
    void PDB2Coords( const char *filename, THDoubleTensor *coords ){
        if(coords->nDimension == 1){
            std::string f(filename);
            cPDBLoader pdb(f);
            pdb.reorder(THDoubleTensor_data(coords));
        }else if(coords->nDimension == 2){
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