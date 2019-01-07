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
    std::vector<std::string> chain_names;
    std::vector<int> res_nums;

    //ordering accoring to residues
    std::vector<std::string> res_res_names;
    std::vector<std::vector<cVector3> > res_r;
    std::vector<std::vector<std::string> > res_atom_names;

	cVector3 b0, b1;
public:
    cPDBLoader();
	cPDBLoader(std::string filename);
	virtual ~cPDBLoader();
  
    //order according to cConformation
    void reorder();

    //order according to atom types
    // void reorder(double *coords, int *num_atoms_of_type, int *offsets);

    cVector3 getCenterMass();
    void translate(cVector3 dr);
    // void randRot(THGenerator *gen);
    // void randTrans(THGenerator *gen, int spatial_dim);
    
    inline uint getNumAtoms(){return r.size();};
    void computeBoundingBox();
    


};

#endif /* CPROTEINLOADER_H_ */
