#ifndef CPDBLOADER_H_
#define CPDBLOADER_H_
#include <string>
#include <vector>
#include <cVector3.h>
#include <cMatrix33.h>
#include <TH/TH.h>

class cPDBLoader {
public:
	
	//ordering accoring to PDB lines
	std::vector<cVector3> r;
    std::vector<std::string> atom_names;
    std::vector<std::string> res_names;
    std::vector<int> res_nums;

    //ordering accoring to residues
    std::vector<std::string> res_res_names;
    std::vector<std::vector<cVector3> > res_r;
    std::vector<std::vector<std::string> > res_atom_names;

	
public:
    cPDBLoader();
	cPDBLoader(std::string filename);
	virtual ~cPDBLoader();
    
    //order according to cConformation
    void reorder(double *coords, bool add_terminal=false);
    //order according to atom types
    void reorder(double *coords, uint *num_atoms_of_type, uint *offsets);

    cVector3 getCenterMass();
    void translate(cVector3 dr);
    void randRot(THGenerator *gen);
    inline uint getNumAtoms(){return r.size();};


};

#endif /* CPROTEINLOADER_H_ */
