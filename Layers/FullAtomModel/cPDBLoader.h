#ifndef CPDBLOADER_H_
#define CPDBLOADER_H_
#include <string>
#include <vector>
#include <cVector3.h>
#include <cMatrix33.h>

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
    
    void reorder(double *coords, bool add_terminal=false);
	
    bool isHeavyAtom(std::string &atom_name);
    int getAtomIndex(std::string &res_name, std::string &atom_name);
    uint getNumAtoms(std::string &sequence);
};

inline std::string trim(const std::string &s);

#endif /* CPROTEINLOADER_H_ */
