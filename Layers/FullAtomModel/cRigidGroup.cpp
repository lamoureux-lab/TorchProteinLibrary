#include <cRigidGroup.h>

cRigidGroup::cRigidGroup(){

}

cRigidGroup::~cRigidGroup(){
    
}

void cRigidGroup::addAtom(cVector3 pos, std::string atomName){
    this->atoms_local.push_back(pos);
    this->atoms_global.push_back(pos);
    this->atomNames.push_back(atomName);
}

void cRigidGroup::applyTransform(cMatrix44 mat){
    for(int i=0; i<this->atoms_local.size(); i++){
        atoms_global[i] = mat*atoms_local[i];
    }
}

std::ostream& operator<<(std::ostream& os, const cRigidGroup& rg){
    for(int i=0; i<rg.atomNames.size(); i++){
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
cRigidGroup *makeSerGroup(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CB");
    g->addAtom(geo.R_CB_OG*cVector3(cos(-(M_PI-geo.CA_CB_OG_angle)),0,
                        sin(-(M_PI-geo.CA_CB_OG_angle))), "OG");
    return g;
}
cRigidGroup *makeCysGroup(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CB");
    g->addAtom(geo.R_CB_SG*cVector3(cos(-(M_PI-geo.CA_CB_SG_angle)),0,
                        sin(-(M_PI-geo.CA_CB_SG_angle))), "SG");
    return g;
}
cRigidGroup *makeValGroup(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CB");
    g->addAtom(geo.R_CB_CG*cVector3(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.CG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.CG1_CB_CG2_angle/2.0)), "CG1");
    g->addAtom(geo.R_CB_CG*cVector3(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        -cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.CG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.CG1_CB_CG2_angle/2.0)), "CG2");
    return g;
}
