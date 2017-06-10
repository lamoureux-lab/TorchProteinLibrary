/*
 * cProteinLoader.h
 *
 *  Created on: Jun 18, 2016
 *      Author: lupoglaz
 */

#ifndef CPROTEINLOADER_H_
#define CPROTEINLOADER_H_
#include "TH/TH.h"
#include <string>
#include <vector>
#include <cVector3.h>
#include <cMatrix33.h>

class cProteinLoader {
public:
	float res;
	//atom coordinates and types
	std::vector<int> atomType;
	std::vector<cVector3> r;
	std::vector<std::string> lines;
	std::vector<cVector3> dr;
	std::vector<int> line_num;
	std::string filename;
	//bounding box
	cVector3 b0, b1;
	int num_atom_types;

public:
	cProteinLoader();
	virtual ~cProteinLoader();
	void print(int num);

	int loadPDB(std::string filename);
	int savePDB(std::string filename);
	
	int assignAtomTypes(int assigner_type); // 1 - 4 atom types; 2 - 10 atom types
	void projectToTensor(THFloatTensor *grid);

	void save_binary(std::string filename);
	void load_binary(std::string filename);

	void computeBoundingBox();

	void shiftProtein(cVector3 dr);
	void rotateProtein(cMatrix33 rot);
	
	void addToGridExp(int aInd, THFloatTensor *grid);
	void addToGridBin(int aInd, THFloatTensor *grid);

	int get11AtomType(std::string res_name, std::string atom_name, bool terminal);
	int get4AtomType(std::string &atom_name);
	
	THGenerator *gen;

};

#endif /* CPROTEINLOADER_H_ */
