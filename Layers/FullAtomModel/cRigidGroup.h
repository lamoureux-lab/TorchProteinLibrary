#ifndef CRIGIDGROUP_H_
#define CRIGIDGROUP_H_
#include <cVector3.h>
#include <cMatrix44.h>
#include <vector>
#include <iostream>
#include <cGeometry.h>

class cRigidGroup{
    // private:
    //     std::vector<cVector3> atoms;
    public:
        std::vector<cVector3> atoms_local;
        std::vector<cVector3> atoms_global;
        std::vector<std::string> atomNames;
        cRigidGroup();
        ~cRigidGroup();
        void addAtom(cVector3 pos, std::string atomName);
        void applyTransform(cMatrix44 mat);
};

cRigidGroup *makeAtom(std::string atomName);
cRigidGroup *makeCarbonyl(cGeometry &geo);
std::ostream& operator<<(std::ostream& os, const cRigidGroup& rg);


#endif