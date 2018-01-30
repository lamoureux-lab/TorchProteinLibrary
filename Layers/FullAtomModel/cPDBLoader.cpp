/*
 * cProteinLoader.cpp
 *
 *  Created on: Jun 18, 2016
 *      Author: lupoglaz
 */

#include <iostream>
#include <fstream>
#include <exception>
#include <math.h>
#include <string>
#include <cctype>
#include <string>
#include <algorithm>
#include "cPDBLoader.h"

cPDBLoader::cPDBLoader(){

}
cPDBLoader::cPDBLoader(std::string filename) {
	std::ifstream pfile(filename);
	std::string line, header, xStr, yStr, zStr, atom_name, res_name;
    int res_num;

	// reading raw file
	while ( getline (pfile,line) ){
		header = line.substr(0,4);
        atom_name = trim(line.substr(13,4));
        
        if( header.compare("ATOM")==0 && isHeavyAtom(atom_name)){
            xStr = line.substr(30,8);
			yStr = line.substr(38,8);
			zStr = line.substr(46,8);
            res_name = trim(line.substr(17,3));
		    res_num = std::stoi(line.substr(23,4));
			r.push_back(cVector3(std::stof(xStr),std::stof(yStr),std::stof(zStr)));
            res_names.push_back(res_name);
            res_nums.push_back(res_num);
            atom_names.push_back(atom_name);
		}
	}
    
    // reorganizing according to residue numbers
    int num_residues = res_nums.back() - res_nums[0] + 1;
    res_r.resize(num_residues);
    res_atom_names.resize(num_residues);
    res_res_names.resize(num_residues);
    for(int i=0; i<r.size(); i++){
        res_num = res_nums[i] - res_nums[0];
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
    std::string lastO("O");
    // reordering atoms according to cConformation output
    for(int i=0; i<res_r.size(); i++){
        for(int j=0; j<res_r[i].size(); j++){
            local_ind = getAtomIndex(res_res_names[i], res_atom_names[i][j]);
            if(local_ind == -1){
                throw std::string("cPDBLoader::reorder: unknown atom ") + res_res_names[i] +std::string(":")+res_atom_names[i][j];
            }
            cVector3 global_r(coords + (global_ind + local_ind)*3);
            global_r = res_r[i][j];
        }
        if( i<(res_r.size()-1) )
            lastO = "O";
        else
            lastO = "OXT";
        if(res_r[i].size()!= (getAtomIndex(res_res_names[i], lastO) + 1) ){
            throw std::string("cPDBLoader::reorder: Missing atoms");
        }
        global_ind += getAtomIndex(res_res_names[i], lastO) + 1;
    }       
}

bool cPDBLoader::isHeavyAtom(std::string &atom_name){
    if(atom_name[0] == 'C' || atom_name[0] == 'N' || atom_name[0] == 'O' || atom_name[0] == 'S')
        return true;
    else
        return false;
}

uint cPDBLoader::getNumAtoms(std::string &sequence){
    uint num_atoms = 0;
    std::string lastO("O");
    for(int i=0; i<sequence.length(); i++){
        std::string AA(1, sequence[i]);
        if( i<(sequence.length()-1) )
            lastO = "O";
        else
            lastO = "OXT";
        num_atoms += getAtomIndex(AA, lastO) + 1;
    }
    return num_atoms;
}

int cPDBLoader::getAtomIndex(std::string &res_name, std::string &atom_name){
    if(atom_name == std::string("N"))
        return 0;
    if(atom_name == std::string("CA"))
        return 1;
    if(atom_name == std::string("CB"))
        return 2;

    if(res_name == std::string("GLY") || res_name == std::string("G")){
        if(atom_name == std::string("C"))
            return 2;
        if(atom_name == std::string("O"))
            return 3;
        if(atom_name == std::string("OXT"))
            return 4;
    }

    if(res_name == std::string("ALA") || res_name == std::string("A")){
        if(atom_name == std::string("C"))
            return 3;
        if(atom_name == std::string("O"))
            return 4;
        if(atom_name == std::string("OXT"))
            return 5;
    }

    if(res_name == std::string("SER") || res_name == std::string("S")){
        if(atom_name == std::string("OG"))
            return 3;
        if(atom_name == std::string("C"))
            return 4;
        if(atom_name == std::string("O"))
            return 5;
        if(atom_name == std::string("OXT"))
            return 5;
    }

    if(res_name == std::string("CYS") || res_name == std::string("C")){
        if(atom_name == std::string("SG"))
            return 3;
        if(atom_name == std::string("C"))
            return 4;
        if(atom_name == std::string("O"))
            return 5;
        if(atom_name == std::string("OXT"))
            return 6;
    }

    if(res_name == std::string("VAL") || res_name == std::string("V")){
        if(atom_name == std::string("CG1"))
            return 3;
        if(atom_name == std::string("CG2"))
            return 4;
        if(atom_name == std::string("C"))
            return 5;
        if(atom_name == std::string("O"))
            return 6;
        if(atom_name == std::string("OXT"))
            return 7;
    }

    if(res_name == std::string("ILE") || res_name == std::string("I")){
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
        if(atom_name == std::string("OXT"))
            return 8;
    }

    if(res_name == std::string("LEU") || res_name == std::string("L")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD1"))
            return 4;
        if(atom_name == std::string("CD2"))
            return 5;
        if(atom_name == std::string("C"))
            return 6;
        if(atom_name == std::string("O"))
            return 7;
        if(atom_name == std::string("OXT"))
            return 8;
    }

    if(res_name == std::string("THR") || res_name == std::string("T")){
        if(atom_name == std::string("OG1"))
            return 3;
        if(atom_name == std::string("CG2"))
            return 4;
        if(atom_name == std::string("C"))
            return 5;
        if(atom_name == std::string("O"))
            return 6;
        if(atom_name == std::string("OXT"))
            return 7;
    }

    if(res_name == std::string("ARG") || res_name == std::string("R")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD"))
            return 4;
        if(atom_name == std::string("NE"))
            return 5;
        if(atom_name == std::string("CZ"))
            return 6;
        if(atom_name == std::string("NH1"))
            return 7;
        if(atom_name == std::string("NH2"))
            return 8;
        if(atom_name == std::string("C"))
            return 9;
        if(atom_name == std::string("O"))
            return 10;
        if(atom_name == std::string("OXT"))
            return 11;
    }

    if(res_name == std::string("LYS") || res_name == std::string("K")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD"))
            return 4;
        if(atom_name == std::string("CE"))
            return 5;
        if(atom_name == std::string("NZ"))
            return 6;
        if(atom_name == std::string("C"))
            return 7;
        if(atom_name == std::string("O"))
            return 8;
        if(atom_name == std::string("OXT"))
            return 9;
    }

    if(res_name == std::string("ASP") || res_name == std::string("D")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("OD1"))
            return 4;
        if(atom_name == std::string("OD2"))
            return 5;
        if(atom_name == std::string("C"))
            return 6;
        if(atom_name == std::string("O"))
            return 7;
        if(atom_name == std::string("OXT"))
            return 8;
    }

    if(res_name == std::string("ASN") || res_name == std::string("N")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("OD1"))
            return 4;
        if(atom_name == std::string("ND2"))
            return 5;
        if(atom_name == std::string("C"))
            return 6;
        if(atom_name == std::string("O"))
            return 7;
        if(atom_name == std::string("OXT"))
            return 8;
    }

    if(res_name == std::string("GLU") || res_name == std::string("E")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD"))
            return 4;
        if(atom_name == std::string("OE1"))
            return 5;
        if(atom_name == std::string("OE2"))
            return 6;
        if(atom_name == std::string("C"))
            return 7;
        if(atom_name == std::string("O"))
            return 8;
        if(atom_name == std::string("OXT"))
            return 9;
    }

    if(res_name == std::string("GLN") || res_name == std::string("Q")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD"))
            return 4;
        if(atom_name == std::string("OE1"))
            return 5;
        if(atom_name == std::string("NE2"))
            return 6;
        if(atom_name == std::string("C"))
            return 7;
        if(atom_name == std::string("O"))
            return 8;
        if(atom_name == std::string("OXT"))
            return 9;
    }

    if(res_name == std::string("MET") || res_name == std::string("M")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("SD"))
            return 4;
        if(atom_name == std::string("CE"))
            return 5;
        if(atom_name == std::string("C"))
            return 6;
        if(atom_name == std::string("O"))
            return 7;
        if(atom_name == std::string("OXT"))
            return 8;
    }

    if(res_name == std::string("HIS") || res_name == std::string("H")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD2"))
            return 4;
        if(atom_name == std::string("NE2"))
            return 5;
         if(atom_name == std::string("CE1"))
            return 6;
        if(atom_name == std::string("ND1"))
            return 7;
        if(atom_name == std::string("C"))
            return 8;
        if(atom_name == std::string("O"))
            return 9;
        if(atom_name == std::string("OXT"))
            return 10;
    }

    if(res_name == std::string("PRO") || res_name == std::string("P")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD"))
            return 4;
        if(atom_name == std::string("C"))
            return 5;
        if(atom_name == std::string("O"))
            return 6;
        if(atom_name == std::string("OXT"))
            return 7;
    }

    if(res_name == std::string("PHE") || res_name == std::string("F")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD1"))
            return 4;
        if(atom_name == std::string("CE1"))
            return 5;
         if(atom_name == std::string("CZ"))
            return 6;
        if(atom_name == std::string("CE2"))
            return 7;
        if(atom_name == std::string("CD2"))
            return 8;
        if(atom_name == std::string("C"))
            return 9;
        if(atom_name == std::string("O"))
            return 10;
        if(atom_name == std::string("OXT"))
            return 11;
    }

    if(res_name == std::string("TYR") || res_name == std::string("Y")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD1"))
            return 4;
        if(atom_name == std::string("CE1"))
            return 5;
         if(atom_name == std::string("CZ"))
            return 6;
        if(atom_name == std::string("CE2"))
            return 7;
        if(atom_name == std::string("CD2"))
            return 8;
        if(atom_name == std::string("OH"))
            return 9;
        if(atom_name == std::string("C"))
            return 10;
        if(atom_name == std::string("O"))
            return 11;
        if(atom_name == std::string("OXT"))
            return 12;
    }

    if(res_name == std::string("TRP") || res_name == std::string("W")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD1"))
            return 4;
        if(atom_name == std::string("CD2"))
            return 5;
         if(atom_name == std::string("NE1"))
            return 6;
        if(atom_name == std::string("CE2"))
            return 7;
        if(atom_name == std::string("CE3"))
            return 8;
        if(atom_name == std::string("CZ2"))
            return 9;
        if(atom_name == std::string("CZ3"))
            return 10;
        if(atom_name == std::string("CH2"))
            return 11;
        if(atom_name == std::string("C"))
            return 12;
        if(atom_name == std::string("O"))
            return 13;
        if(atom_name == std::string("OXT"))
            return 14;
    }

    return -1;
}

inline std::string trim(const std::string &s)
{
   auto wsfront=std::find_if_not(s.begin(),s.end(),[](int c){return std::isspace(c);});
   auto wsback=std::find_if_not(s.rbegin(),s.rend(),[](int c){return std::isspace(c);}).base();
   return (wsback<=wsfront ? std::string() : std::string(wsfront,wsback));
}