#include <TH/TH.h>
#include "cPDBLoader.h"
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
    int getSeqNumAtoms( const char *sequence){
        cPDBLoader pdb;
        std::string seq(sequence);
        int num_atoms = pdb.getNumAtoms(seq);
        return num_atoms;
    }
}