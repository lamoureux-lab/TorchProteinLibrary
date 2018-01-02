#include <cRigidGroup.h>

cRigidGroup::cRigidGroup(){

}

cRigidGroup::~cRigidGroup(){
    
}

void cRigidGroup::addAtom(cVector3 pos, std::string atomName){
    this->atoms.push_back(pos);
    this->atomNames.push_back(atomName);
}

void cRigidGroup::applyTransform(cMatrix44 mat){
    for(int i=0; i<this->atoms.size(); i++){
        atoms[i] = mat*atoms[i];
    }
}

std::ostream& operator<<(std::ostream& os, const cRigidGroup& rg){
    for(int i=0; i<rg.atoms.size(); i++){
        os<<rg.atomNames[i];
    }
    return os;
}

cRigidGroup *makeAtom(std::string atomName){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), atomName);
    return g;
}

cRigidGroup *makeCarbonyl(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "C");
    g->addAtom(cVector3(cos(geo.CA_C_O_angle),0,sin(geo.CA_C_O_angle)), "O");
    return g;
}
