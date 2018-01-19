#ifndef CRIGIDGROUP_H_
#define CRIGIDGROUP_H_
#include <cVector3.h>
#include <cMatrix44.h>
#include <vector>
#include <iostream>
#include <cGeometry.h>

class cRigidGroup{
    public:
        std::vector<cVector3> atoms_local;
        std::vector<cVector3> atoms_global;
        std::vector<std::string> atomNames;
        std::vector<uint> atomIndexes;
        uint residueIndex;
        char residueName;
        cRigidGroup();
        ~cRigidGroup();
        void addAtom(cVector3 pos, std::string atomName, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
        void applyTransform(cMatrix44 mat);
};

cRigidGroup *makeAtom(std::string atomName, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeCarbonyl(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeSerGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeCysGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeValGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeIleGroup1(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeIleGroup2(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeLeuGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeThrGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeArgGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeAspGroup(cGeometry &geo, std::string O1, std::string O2, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeAsnGroup(cGeometry &geo, std::string O1, std::string N2, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeHisGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeProGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makePheGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeTyrGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
cRigidGroup *makeTrpGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr);
std::ostream& operator<<(std::ostream& os, const cRigidGroup& rg);


#endif