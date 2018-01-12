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
cRigidGroup *makeSerGroup(cGeometry &geo);
cRigidGroup *makeCysGroup(cGeometry &geo);
cRigidGroup *makeValGroup(cGeometry &geo);
cRigidGroup *makeIleGroup1(cGeometry &geo);
cRigidGroup *makeIleGroup2(cGeometry &geo);
cRigidGroup *makeLeuGroup(cGeometry &geo);
cRigidGroup *makeThrGroup(cGeometry &geo);
cRigidGroup *makeArgGroup(cGeometry &geo);
cRigidGroup *makeAspGroup(cGeometry &geo, std::string O1, std::string O2);
cRigidGroup *makeAsnGroup(cGeometry &geo, std::string O1, std::string N2);
cRigidGroup *makeHisGroup(cGeometry &geo);
cRigidGroup *makeProGroup(cGeometry &geo);
cRigidGroup *makePheGroup(cGeometry &geo);
cRigidGroup *makeTyrGroup(cGeometry &geo);
cRigidGroup *makeTrpGroup(cGeometry &geo);
std::ostream& operator<<(std::ostream& os, const cRigidGroup& rg);


#endif