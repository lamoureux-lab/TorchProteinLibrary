#include <cRigidGroup.h>

cRigidGroup::cRigidGroup(){

}

cRigidGroup::~cRigidGroup(){
    
}

void cRigidGroup::addAtom(cVector3 &pos){
    this->atoms.push_back(pos);
}

void cRigidGroup::applyTransform(cMatrix44 &mat){
    for(int i=0; i<this->atoms.size(); i++){
        atoms[i] = mat*atoms[i];
    }
}

cRigidGroup makeAtom(){
    cRigidGroup g;
    g.addAtom(cVector3(0,0,0));
    return g;
}
