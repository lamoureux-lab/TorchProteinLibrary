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
#include <nUtil.h>

using namespace ProtUtil;
using namespace StringUtil;

cPDBLoader::cPDBLoader(){

}
cPDBLoader::cPDBLoader(std::string filename) {
    std::ifstream pfile(filename);
	std::string line, header, xStr, yStr, zStr, atom_name, res_name;
    int res_num;
	// reading raw file
	while ( getline (pfile,line) ){
		header = line.substr(0,4);
                
        if( header.compare("ATOM")==0){
            atom_name = trim(line.substr(13,4));
            if(isHeavyAtom(atom_name)){
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

void cPDBLoader::reorder(double *coords, bool add_terminal){
    int global_ind=0;
    int local_ind;
    std::string lastO("O");
    // reordering atoms according to cConformation output
    for(int i=0; i<res_r.size(); i++){
        for(int j=0; j<res_r[i].size(); j++){
            local_ind = getAtomIndex(res_res_names[i], res_atom_names[i][j]);
            if(local_ind == -1){
                std::cout<<"cPDBLoader::reorder: unknown atom "<<std::endl;
                throw std::string("cPDBLoader::reorder: unknown atom ") + res_res_names[i] +std::string(":")+res_atom_names[i][j];
            }
            cVector3 global_r(coords + (global_ind + local_ind)*3);
            global_r = res_r[i][j];
        }
        if(add_terminal){
            if( i<(res_r.size()-1) )
                lastO = "O";
            else
                lastO = "OXT";
        }else{
            lastO = "O";
        }
        if(res_r[i].size()!= (getAtomIndex(res_res_names[i], lastO) + 1) ){
            std::cout<<"Missing atoms"<<std::endl;
            throw std::string("cPDBLoader::reorder: Missing atoms");
        }
        global_ind += getAtomIndex(res_res_names[i], lastO) + 1;
    }       
}

cVector3 cPDBLoader::getCenterMass(){
    cVector3 c(0., 0., 0.);
    for(int i=0; i<r.size(); i++){
        c += r[i];
    }
    c /= double(r.size());
    return c;
}

void cPDBLoader::translate(cVector3 dr){
    for(int i=0; i<r.size(); i++){
        r[i] += dr;
    }
}

void cPDBLoader::computeBoundingBox(){
	b0[0]=std::numeric_limits<float>::infinity(); 
	b0[1]=std::numeric_limits<float>::infinity(); 
	b0[2]=std::numeric_limits<float>::infinity();
	b1[0]=-1*std::numeric_limits<float>::infinity(); 
	b1[1]=-1*std::numeric_limits<float>::infinity(); 
	b1[2]=-1*std::numeric_limits<float>::infinity();
	for(int i=0;i<r.size();i++){
		if(r[i][0]<b0[0]){
			b0[0]=r[i][0];
		}
		if(r[i][1]<b0[1]){
			b0[1]=r[i][1];
		}
		if(r[i][2]<b0[2]){
			b0[2]=r[i][2];
		}

		if(r[i][0]>b1[0]){
			b1[0]=r[i][0];
		}
		if(r[i][1]>b1[1]){
			b1[1]=r[i][1];
		}
		if(r[i][2]>b1[2]){
			b1[2]=r[i][2];
		}
	}
}

void cPDBLoader::randRot(THGenerator *gen){
    float u1 = THRandom_uniform(gen,0,1.0);
    float u2 = THRandom_uniform(gen,0,1.0);
    float u3 = THRandom_uniform(gen,0,1.0);
    float q[4];
    q[0] = sqrt(1-u1) * sin(2.0*M_PI*u2);
    q[1] = sqrt(1-u1) * cos(2.0*M_PI*u2);
    q[2] = sqrt(u1) * sin(2.0*M_PI*u3);
    q[3] = sqrt(u1) * cos(2.0*M_PI*u3);
    cMatrix33 random_rotation;
    random_rotation.m[0][0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
    random_rotation.m[0][1] = 2.0*(q[1]*q[2] - q[0]*q[3]);
    random_rotation.m[0][2] = 2.0*(q[1]*q[3] + q[0]*q[2]);

    random_rotation.m[1][0] = 2.0*(q[1]*q[2] + q[0]*q[3]);
    random_rotation.m[1][1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
    random_rotation.m[1][2] = 2.0*(q[2]*q[3] - q[0]*q[1]);

    random_rotation.m[2][0] = 2.0*(q[1]*q[3] - q[0]*q[2]);
    random_rotation.m[2][1] = 2.0*(q[2]*q[3] + q[0]*q[1]);
    random_rotation.m[2][2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
    
    for(int i=0; i<r.size(); i++){
        r[i] = random_rotation*r[i];
    }
}
void cPDBLoader::randTrans(THGenerator *gen, int spatial_dim){
    float dx_max = fmax(0, spatial_dim/2.0 - (b1[0]-b0[0])/2.0)*0.5;
    float dy_max = fmax(0, spatial_dim/2.0 - (b1[1]-b0[1])/2.0)*0.5;
    float dz_max = fmax(0, spatial_dim/2.0 - (b1[2]-b0[2])/2.0)*0.5;
    float dx = THRandom_uniform(gen,-dx_max,dx_max);
    float dy = THRandom_uniform(gen,-dy_max,dy_max);
    float dz = THRandom_uniform(gen,-dz_max,dz_max);
    this->translate(cVector3(dx,dy,dz));
}
void cPDBLoader::reorder(double *coords, uint *num_atoms_of_type, uint *offsets){
    std::vector<uint> atom_types;
    uint num_atoms[11];
    for(int i=0;i<11;i++){
        num_atoms_of_type[i] = 0;
        num_atoms[i] = 0;
        offsets[i] = 0;
    }

    bool terminal = false;
    for(int i=0; i<r.size(); i++){
        uint type;
        if(res_nums[i] == res_res_names.size())
            terminal = true;
        type = get11AtomType(res_names[i], atom_names[i], terminal);
        atom_types.push_back(type);
        num_atoms_of_type[type]+=1;
    }
    for(int i=1;i<11;i++){
        offsets[i] = offsets[i-1] + num_atoms_of_type[i-1];
    }

    for(int i=0; i<r.size(); i++){
        uint type = atom_types[i];
        cVector3 r_target(coords + 3*(offsets[type] + num_atoms[type]));
        r_target = r[i];
        num_atoms[type]+=1;
    }
}