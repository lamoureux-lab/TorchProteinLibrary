#ifndef CRIGIDGROUP_H_
#define CRIGIDGROUP_H_
#include <cVector3.h>
#include <cVector44.h>
#include <vector>

// template <class T>
// class  cTreeNode{
//     private:
//         int num;
//         cTreeNode *left;
//         cTreeNode *right;
//     public:
//         cTreeNode();
//         cTreeNode(num);
//         friend class cTree;
// }

// template <class T>
// class cTree{

// };

class cConformation{
    std::vector<cGroup> groups;
    std::vector<cMatrix44> transforms;
	
public:

	cConformation();
    ~cConformation();
    void addGroup(cRigidGroup group);


    
};