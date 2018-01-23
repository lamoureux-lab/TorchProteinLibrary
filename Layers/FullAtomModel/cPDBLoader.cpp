/*
 * cProteinLoader.cpp
 *
 *  Created on: Jun 18, 2016
 *      Author: lupoglaz
 */

#include <iostream>
#include <fstream>
#include <exception>
#include <limits>
#include <math.h>
#include <string>
#include "cProteinLoader.h"
#include <algorithm>
#include <memory>
#include <cstdio>

cPDBLoader::cPDBLoader(std::string filename) {
	std::ifstream pfile(filename);
	std::string line, header, xStr, yStr, zStr, atom_name, res_name;
    int res_num;

	// reading raw file
	while ( getline (pfile,line) ){
		header = line.substr(0,4);
        atom_name = line.substr(13,4);
        if( header.compare("ATOM")==0 && isHeavyAtom(atom_name)){
            xStr = line.substr(30,8);
			yStr = line.substr(38,8);
			zStr = line.substr(46,8);
            res_name = line.substr(17,3);
		    res_num = std::stoi(line.substr(23,4));
			r.push_back(cVector3(std::stof(xStr),std::stof(yStr),std::stof(zStr)));
            res_names.push_back(res_name);
            res_nums.push_back(res_num);
		}
	}

    // reorganizing according to residue numbers
    int num_residues = res_nums.back() - res_nums[0];
    res_r.resize(num_resid);
    res_atom_names.resize(num_resid);
    for(int i=0; i<r.size(); i++){
        res_num = res_nums[i];
        res_r[res_num].push_back(r[i]);
        res_atom_names[res_num].push_back(atom_names[i]);
        res_res_names[res_num] = res_names[i];
    }
}

cPDBLoader::~cPDBLoader() {
		
}

void cPDBLoader::reorder(double *coords){
    int global_ind=0;
    int local_ind;
    // reordering atoms according to cConformation output
    for(int i=0; i<res_r.size(); i++){
        for(int j=0; j<res_r[i].size(); j++){
            local_ind = getAtomIndex(res_res_names[i], res_atom_names[i][j]);
            cVector3 global_r(coords + (global_ind + local_ind)*3);
            global_r = res_r[i];
        }
        global_ind += res_r[i].size();
    }       
}

bool cPDBLoader::isHeavyAtom(std::string &atom_name){
    if(atom_name[0] == 'C' || atom_name[0] == 'N' || atom_name[0] == 'O' || atom_name[0] == 'S')
        return true;
    else
        return false;
}
int cPDBLoader::getAtomIndex(std::string &res_name, std::string &atom_name){
    if(atom_name == std::string("N"))
        return 0;
    if(atom_name == std::string("CA"))
        return 1;
    if(atom_name == std::string("CB"))
        return 2;

    if(res_name == std::string("GLY")){
        if(atom_name == std::string("C"))
            return 2;
        if(atom_name == std::string("O"))
            return 3;
    }

    if(res_name == std::string("ALA")){
        if(atom_name == std::string("C"))
            return 3;
        if(atom_name == std::string("O"))
            return 4;
    }

    if(res_name == std::string("SER")){
        if(atom_name == std::string("OG"))
            return 3;
        if(atom_name == std::string("C"))
            return 4;
        if(atom_name == std::string("O"))
            return 5;
    }

    if(res_name == std::string("CYS")){
        if(atom_name == std::string("SG"))
            return 3;
        if(atom_name == std::string("C"))
            return 4;
        if(atom_name == std::string("O"))
            return 5;
    }

    if(res_name == std::string("VAL")){
        if(atom_name == std::string("CG1"))
            return 3;
        if(atom_name == std::string("CG2"))
            return 4;
        if(atom_name == std::string("C"))
            return 5;
        if(atom_name == std::string("O"))
            return 6;
    }

    if(res_name == std::string("ILE")){
        if(atom_name == std::string("CG2"))
            return 3;
        if(atom_name == std::string("CG1"))
            return 4;
        if(atom_name == std::string("CD1"))
            return 5;
        if(atom_name == std::string("C"))
            return 6;
        if(atom_name == std::string("O"))
            return 7;
    }

    if(res_name == std::string("LEU")){
        if(atom_name == std::string("CG1"))
            return 3;
        if(atom_name == std::string("CD1"))
            return 4;
        if(atom_name == std::string("CD2"))
            return 5;
        if(atom_name == std::string("C"))
            return 6;
        if(atom_name == std::string("O"))
            return 7;
    }

    if(res_name == std::string("THR")){
        if(atom_name == std::string("OG1"))
            return 3;
        if(atom_name == std::string("CG2"))
            return 4;
        if(atom_name == std::string("C"))
            return 5;
        if(atom_name == std::string("O"))
            return 6;
    }

    if(res_name == std::string("ARG")){
        if(atom_name == std::string("CG1"))
            return 3;
        if(atom_name == std::string("CD1"))
            return 4;
        if(atom_name == std::string("CD2"))
            return 5;
        if(atom_name == std::string("C"))
            return 6;
        if(atom_name == std::string("O"))
            return 7;
    }

}

