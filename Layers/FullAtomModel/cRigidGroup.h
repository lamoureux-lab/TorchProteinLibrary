#ifndef CRIGIDGROUP_H_
#define CRIGIDGROUP_H_
#include <cVector3.h>
#include <cVector44.h>
#include <vector>


class cRigidGroup{
    std::vector<cVector3> atoms;
	
public:

	cRigidGroup();
    ~cRigidGroup();
    void addAtom(cVector3 &pos);
    void applyTransform(cMatrix44 &mat);

};

cRigidGroup makeAtom(){
    cRigidGroup g;
    g.addAtom(cVector3(0,0,0));
    return g;
}

cRigidGroup makeRing(){
    cRigidGroup g;
    g.addAtom(cVector3(0,0,0));
    return g;
}


#endif