#ifndef CRIGIDGROUP_H_
#define CRIGIDGROUP_H_
#include <cVector3.h>
#include <cVector44.h>
#include <vector>


class cRigidGroup{
    std::vector<cVector3*> group;
	
public:

	cRigidGroup();
    ~cRigidGroup();

};

#endif