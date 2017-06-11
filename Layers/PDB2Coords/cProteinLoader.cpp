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

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf( new char[ size ] ); 
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

cProteinLoader::cProteinLoader() {
	num_atom_types = -1;
	gen = THGenerator_new();
 	THRandom_seed(gen);
}

cProteinLoader::~cProteinLoader() {
	// TODO Auto-generated destructor stub
	THGenerator_free(gen);
}

void cProteinLoader::pdb2Coords(THCState *state,
								std::string filename, 
								THCudaTensor *gpu_plain_coords, 
								THCudaIntTensor *gpu_offsets, 
								THCudaIntTensor *gpu_num_coords_of_type,
								int spatial_dim,
								int resolution,
								bool rotate,
								bool shift){
	this->loadPDB(filename);
	this->assignAtomTypes(2);
	this->computeBoundingBox();
	this->shiftProtein( -0.5*(b0 + b1) );
	if(rotate){
		cVector3 uniform;
		cMatrix33 rotation;
		uniform.makeUniformVector(gen);
		rotation.makeUniformRotation(uniform);
		rotateProtein(rotation);
	}
	if(shift){
		float dx_max = fmax(0, spatial_dim*resolution/2.0 - (b1[0]-b0[0])/2.0)*0.5;
		float dy_max = fmax(0, spatial_dim*resolution/2.0 - (b1[1]-b0[1])/2.0)*0.5;
		float dz_max = fmax(0, spatial_dim*resolution/2.0 - (b1[2]-b0[2])/2.0)*0.5;
		float dx = THRandom_uniform(gen,-dx_max,dx_max);
		float dy = THRandom_uniform(gen,-dy_max,dy_max);
		float dz = THRandom_uniform(gen,-dz_max,dz_max);
		shiftProtein(cVector3(dx,dy,dz));
	}
	this->shiftProtein( 0.5*cVector3(spatial_dim, spatial_dim, spatial_dim)*resolution ); 
	
	//contructing arrays for atom coordinates and types
	std::vector<std::vector<float>> coords(num_atom_types, std::vector<float>(0));
	
	for(int i=0; i<r.size(); i++){
		coords[atomType[i]].push_back(r[i].v[0]);
		coords[atomType[i]].push_back(r[i].v[1]);
		coords[atomType[i]].push_back(r[i].v[2]);
	}

	int plain_coords_size = r.size()*3;	
	float *cpu_plain_coords = new float[plain_coords_size];
	int *cpu_offsets = new int[num_atom_types];
	int *cpu_num_coords_of_type = new int[num_atom_types];
	cpu_offsets[0] = 0;
	for(int i=0; i<num_atom_types;i++){
		cpu_num_coords_of_type[i] = coords[i].size();
		std:copy(coords[i].begin(), coords[i].end(), cpu_plain_coords+cpu_offsets[i]);
		
		if(i<(num_atom_types-1))
			cpu_offsets[i+1] = cpu_offsets[i] + coords[i].size();
	}
	
	cudaMemcpy( THCudaIntTensor_data(state, gpu_offsets), cpu_offsets, num_atom_types*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( THCudaIntTensor_data(state, gpu_num_coords_of_type), cpu_num_coords_of_type, num_atom_types*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( THCudaTensor_data(state, gpu_plain_coords), cpu_plain_coords, plain_coords_size*sizeof(float), cudaMemcpyHostToDevice);

	delete [] cpu_plain_coords;
	delete [] cpu_offsets;
	delete [] cpu_num_coords_of_type;

}

int cProteinLoader::loadPDB(std::string filename){
	std::ifstream pfile(filename);
	std::string line, header, strAtomType, coordsLine, xStr, yStr, zStr;
	std::string atom_name, res_name;
	std::exception p;

	this->filename = filename;

	r.resize(0);
	atomType.resize(0);
	lines.resize(0);

	while ( getline (pfile,line) ){
		header = line.substr(0,4);
		if( header.compare("ATOM")==0){ //if it is an atom record
			lines.push_back(line);
		}
	}
	return 0;
}

int cProteinLoader::savePDB(std::string filename){
	std::ofstream pfile(filename, std::ofstream::out | std::ofstream::app);
	if(dr.size() == r.size()){
		for(int i=0; i<r.size(); i++){
			pfile<<lines[line_num[i]].substr(0, lines[line_num[i]].length()-1)+string_format("%6.2f%6.2f%6.2f\n",dr[i].v[0],dr[i].v[1],dr[i].v[2]);
		}
	}else{
		pfile<<"MODEL\n";
		for(int i=0; i<r.size(); i++){
			pfile<<lines[line_num[i]].substr(0, 30)+string_format("%8.3f%8.3f%8.3f\n",r[i].v[0],r[i].v[1],r[i].v[2]);
		}
		pfile<<"ENDMDL\n";
	}
	pfile.close();
	return 0;
	
}

int cProteinLoader::assignAtomTypes(int assigner_type){
	std::string xStr, yStr, zStr, atom_name, res_name;
	int atom_type;
	//getting terminal residue number
	int term_res_num = std::stoi(lines[lines.size()-1].substr(23,4));

	for(int i=0;i<lines.size();i++){
	
		if( assigner_type == 1){
			atom_name = lines[i].substr(13,4);
			atom_type = get4AtomType(atom_name);
			if(atom_type<0) continue;
	
		}else if ( assigner_type == 2){
			atom_name = lines[i].substr(13,4);
			res_name = lines[i].substr(17,3);
			int res_num = std::stoi(lines[i].substr(23,4));
			atom_type = get11AtomType(res_name, atom_name, res_num == term_res_num);
			if(atom_type<0) continue;
		
		}else{
			return -1;
		}
		
		try{
			xStr = lines[i].substr(30,8);
			yStr = lines[i].substr(38,8);
			zStr = lines[i].substr(46,8);
			r.push_back(cVector3(std::stof(xStr),std::stof(yStr),std::stof(zStr)));
			line_num.push_back(i);
			atomType.push_back(atom_type);
		}catch(std::exception& e ){
			std::cout<<"Problem in file: "<<filename<<"\n";
			return -1;
		}
	}
	if(assigner_type==1){
		num_atom_types = 4;
	}else if (assigner_type==2){
		num_atom_types = 11;
	}
	return assigner_type;
}


void cProteinLoader::load_binary(std::string filename){
	std::ifstream file;
	file.open(filename, std::ios::in | std::ios::binary);
	int N;
	char * int_buffer = new char [sizeof(int)];
	char * double_buffer = new char [sizeof(double)];
	file.read(int_buffer,sizeof(int));
	N = *((int*)int_buffer);
	file.read(int_buffer,sizeof(int));
	num_atom_types = *((int*)int_buffer);
	
	atomType.resize(0);
	r.resize(0);
	
	for(int i=0; i<N; i++){
		cVector3 r_i;
		int type;
		file.read(double_buffer, sizeof(double));
		r_i.v[0] = *((double*)double_buffer);
		file.read(double_buffer, sizeof(double));
		r_i.v[1] = *((double*)double_buffer);
		file.read(double_buffer, sizeof(double));
		r_i.v[2] = *((double*)double_buffer);
		file.read(int_buffer, sizeof(int));
		type = *((int*)int_buffer);
		atomType.push_back(type);
		r.push_back(r_i);
	}
	file.close();
}

void cProteinLoader::save_binary(std::string filename){
	std::ofstream file;
	file.open(filename, std::ios::out | std::ios::binary);
	int N = r.size();
	file.write((const char*)&N,sizeof(int));
	file.write((const char*)&num_atom_types,sizeof(int));
	for(int i=0; i<N; i++){
		file.write((const char*)&(r[i].v[0]), sizeof(double));
		file.write((const char*)&(r[i].v[1]), sizeof(double));
		file.write((const char*)&(r[i].v[2]), sizeof(double));
		file.write((const char*)&(atomType[i]), sizeof(int));
	}
	file.close();	
}


void cProteinLoader::computeBoundingBox(){
	b0[0]=std::numeric_limits<float>::infinity(); 
	b0[1]=std::numeric_limits<float>::infinity(); 
	b0[2]=std::numeric_limits<float>::infinity();
	b1[0]=-1*std::numeric_limits<float>::infinity(); 
	b1[1]=-1*std::numeric_limits<float>::infinity(); 
	b1[2]=-1*std::numeric_limits<float>::infinity();
	for(int i=0;i<atomType.size();i++){
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

void cProteinLoader::addToGridExp(int aInd, THFloatTensor *grid)
{
	float tmp;
	float val,r2;
	int at = atomType[aInd];
	if(at<0 or at>grid->size[0]){
		std::cout<<"atom type = "<<at<<" Wrong grid dimesion 0 = "<<grid->size[0];
		std::exception p;
		throw p;
	}
	
	//cVector r_new = r[aInd]+gridCenter;
	int ix = floor(r[aInd][0]/res);
	int iy = floor(r[aInd][1]/res);
	int iz = floor(r[aInd][2]/res);
	int d=2;
	for(int i=ix-d; i<=ix+d; i++){
		for(int j=iy-d; j<=iy+d; j++){
			for(int k=iz-d; k<=iz+d; k++){
				if( (i>=0 && i<grid->size[1]) && (j>=0 && j<grid->size[2]) && (k>=0 && k<grid->size[3]) ){
					tmp = THFloatTensor_get4d(grid, at, i, j, k);
					r2 = (r[aInd][0] - i*res)*(r[aInd][0] - i*res)+\
					(r[aInd][1] - j*res)*(r[aInd][1] - j*res)+\
					(r[aInd][2] - k*res)*(r[aInd][2] - k*res);
					val = exp(-r2/2.0);
					THFloatTensor_set4d(grid, at, i, j, k, tmp + val);
				}
			}
		}
	}
}

void cProteinLoader::addToGridBin(int aInd, THFloatTensor *grid)
{
	float tmp;
	int at = atomType[aInd];
	if(at<0 or at>grid->size[0]){
		std::cout<<"atom type = "<<at<<" Wrong grid dimesion 0 = "<<grid->size[0];
		std::exception p;
		throw p;
	}
	int ix = floor(r[aInd][0]/res);
	int iy = floor(r[aInd][1]/res);
	int iz = floor(r[aInd][2]/res);
	if( (ix>=0 && ix<grid->size[1]) && (iy>=0 && iy<grid->size[2]) && (iz>=0 && iz<grid->size[3]) ){
		tmp = THFloatTensor_get4d(grid, at, ix, iy, iz);
		THFloatTensor_set4d(grid, at, ix, iy, iz, tmp + 1.0);
	}
}

void cProteinLoader::projectToTensor(THFloatTensor *grid){
	//float grid_center_x, grid_center_y, grid_center_z;
	if(!(grid->nDimension == 4)){
		std::cout<<"Wrong grid dimesion\n";
		std::exception p;
		throw p;
	}
	THFloatTensor_fill(grid,0.0);
	
	float tmp;
	for(int i=0;i<atomType.size();i++){
		//addToGridBin(i, grid);
		addToGridExp(i, grid);
	}
}

void cProteinLoader::shiftProtein(cVector3 dr){
	for(int i=0;i<atomType.size();i++){
		r[i]+=dr;
	}
	b1 += dr;
	b0 += dr;
}

void cProteinLoader::rotateProtein(cMatrix33 rot){
	for(int i=0;i<atomType.size();i++){
		r[i]=rot*r[i];
	}	
	computeBoundingBox();
}

int cProteinLoader::get11AtomType(std::string res_name, std::string atom_name, bool terminal){
	auto f = [](unsigned char const c) { return std::isspace(c); };
	atom_name.erase(std::remove_if(atom_name.begin(), atom_name.end(), f), atom_name.end());
	int assignedType = 0;
	std::string fullAtomName;

	// dealing with the residue-agnostic atom types
	if(atom_name==std::string("O")){
		if(terminal)
			assignedType = 8;
		else
			assignedType = 6;
	}else if(atom_name==std::string("OXT") and terminal){
			assignedType = 8;
	}else if(atom_name==std::string("OT2") and terminal){
			assignedType = 8;
	}else if(atom_name==std::string("N")){
		assignedType = 2;
	}else if(atom_name==std::string("C")){
		assignedType = 9;
	}else if(atom_name==std::string("CA")){
		assignedType = 11;
	}else{
		// dealing with the residue-dependent atom types
		fullAtomName = res_name + atom_name;
		
		if(fullAtomName == std::string("CYSSG") || fullAtomName == std::string("METSD") || fullAtomName == std::string("MSESE")){
			assignedType = 1;
		}else
		// 2 amide N (original ITScore groups 9 + 10)
		if(fullAtomName == std::string("ASNND2") || fullAtomName == std::string("GLNNE2")){
			assignedType = 2;
		}else
		// 3  Nar Aromatic nitrogens (original ITScore group 11)
		if(fullAtomName == std::string("HISND1") || fullAtomName == std::string("HISNE2") || fullAtomName == std::string("TRPNE1")){
			assignedType = 3;
		}else
		// 4 guanidine N (original ITScore groups 12 +13 )
		if(fullAtomName == std::string("ARGNH1") || fullAtomName == std::string("ARGNH2") || fullAtomName == std::string("ARGNE")){
			assignedType = 4;
		}else
		// 5 N31 Nitrogen with three hydrogens (original ITScore 14)
		if(fullAtomName == std::string("LYSNZ")){
			assignedType = 5;
		}else
		// 6 carboxyl 0 (original ITScore groups 15+16)
		if(fullAtomName == std::string("ACEO") || fullAtomName == std::string("ASNOD1") || fullAtomName == std::string("GLNOE1")){
			assignedType = 6;
		}else
		// 7 O3H Oxygens in hydroxyl groups (original ITScore group 17)
		if(fullAtomName == std::string("SEROG") || fullAtomName == std::string("THROG1") || fullAtomName == std::string("TYROH")){
			assignedType = 7;
		}else
		// 8 O22 Oxygens in carboxyl groups, terminus oxygens (original ITScore group 18)
		if(fullAtomName == std::string("ASPOD1") || fullAtomName == std::string("ASPOD2") || \
			fullAtomName == std::string("GLUOE1") || fullAtomName == std::string("GLUOE2")){
			assignedType = 8;
		}else
		// 9 Sp2 Carbons (groups ITScore 1 + 2 + 3 + 4)
		if(fullAtomName == std::string("ARGCZ") || fullAtomName == std::string("ASPCG") || \
			fullAtomName == std::string("GLUCD") || fullAtomName == std::string("ACEC") || \
			fullAtomName == std::string("ASNCG") || fullAtomName == std::string("GLNCD")){
			assignedType = 9;
		}else
		// 10 Car Aromatic carbons (groups ITScore 5)
		if(fullAtomName == std::string("HISCD2") || fullAtomName == std::string("HISCE1") || \
			fullAtomName == std::string("HISCG") || \
			\
			fullAtomName == std::string("PHECD1") || fullAtomName == std::string("PHECD2") || \
			fullAtomName == std::string("PHECE1") || fullAtomName == std::string("PHECE2")|| \
			fullAtomName == std::string("PHECG") || fullAtomName == std::string("PHECZ") || \
			\
			fullAtomName == std::string("TRPCD1") || fullAtomName == std::string("TRPCD2") || \
			fullAtomName == std::string("TRPCE2") || fullAtomName == std::string("TRPCE3") || \
			fullAtomName == std::string("TRPCG") || fullAtomName == std::string("TRPCH2") || \
			fullAtomName == std::string("TRPCZ2") || fullAtomName == std::string("TRPCZ3") || \
			\
			fullAtomName == std::string("TYRCD1") || fullAtomName == std::string("TYRCD2") || \
			fullAtomName == std::string("TYRCE1") || fullAtomName == std::string("TYRCE2") || \
			fullAtomName == std::string("TYRCG") || fullAtomName == std::string("TYRCZ")){
			assignedType = 10;
		}else
		// 11 Sp3 Carbons (corresponding ITScore groups 6 + 7 + 8)
		if(fullAtomName == std::string("ALACB") || \
			\
			fullAtomName == std::string("ARGCB") || fullAtomName == std::string("ARGCG") || fullAtomName == std::string("ARGCD") || \
			\
			fullAtomName == std::string("ASNCB") || \
			\
			fullAtomName == std::string("ASPCB") || \
			\
			fullAtomName == std::string("GLNCB") || fullAtomName == std::string("GLNCG") || \
			\
			fullAtomName == std::string("GLUCB") || fullAtomName == std::string("GLUCG")|| \
			\
			fullAtomName == std::string("HISCB") || \
			\
			fullAtomName == std::string("ILECB") || fullAtomName == std::string("ILECD1") || \
			fullAtomName == std::string("ILECG1") || fullAtomName == std::string("ILECG2") || \
			\
			fullAtomName == std::string("LEUCB") || fullAtomName == std::string("LEUCD1") || \
			fullAtomName == std::string("LEUCD2") || fullAtomName == std::string("LEUCG") || \
			\
			fullAtomName == std::string("LYSCB") || fullAtomName == std::string("LYSCD") || \
			fullAtomName == std::string("LYSCG") || fullAtomName == std::string("LYSCE") || \
			\
			fullAtomName == std::string("METCB") || fullAtomName == std::string("METCE") || fullAtomName == std::string("METCG") || \
			\
			fullAtomName == std::string("MSECB") || fullAtomName == std::string("MSECE") || fullAtomName == std::string("MSECG") || \
			\
			fullAtomName == std::string("PHECB") || \
			\
			fullAtomName == std::string("PROCB") || fullAtomName == std::string("PROCG") || fullAtomName == std::string("PROCD") || \
			\
			fullAtomName == std::string("SERCB") || \
			\
			fullAtomName == std::string("THRCG2") || \
			\
			fullAtomName == std::string("TRPCB") || \
			\ 
			fullAtomName == std::string("TYRCB") || \
			\
			fullAtomName == std::string("VALCB") || fullAtomName == std::string("VALCG1") || fullAtomName == std::string("VALCG2") || \
			\
			fullAtomName == std::string("ACECH3") || \
			\
			fullAtomName == std::string("THRCB") || \
			\
			fullAtomName == std::string("CYSCB") ){
			assignedType = 11;
		}else{
			assignedType = 0;
		}

	}
	//std::cout<<atom_name<<"|"<<fullAtomName<<" -> "<<assignedType<<"\n";
	return assignedType - 1;
}

int cProteinLoader::get4AtomType(std::string &atom_name){
	std::string strAtomType = atom_name.substr(0,1);
	if( strAtomType.compare("H")==0 ){ // not hydrogen
		return -1;
	}else if( strAtomType.compare("C")==0 ){
		return 0;
	}else if( strAtomType.compare("N")==0 ){
		return 1;
	}else if( strAtomType.compare("O")==0 ){
		return 2;
	}else if( strAtomType.compare("S")==0 ){
		return 3;
	}else{
		return -1;
	}
	return -1;
}