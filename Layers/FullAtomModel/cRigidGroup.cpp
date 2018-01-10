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

cRigidGroup *makeIleGroup1(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CB");
    g->addAtom(geo.R_CB_CG*cVector3(sin(geo.CA_CB_CG1_angle - M_PI/2.0),
                        -cos(geo.CA_CB_CG1_angle - M_PI/2.0)*sin(geo.CG1_CB_CG2_angle),
                        cos(geo.CA_CB_CG1_angle - M_PI/2.0)*cos(geo.CG1_CB_CG2_angle)), "CG2");
    // g->addAtom(geo.R_CB_CG*cVector3(cos(-(M_PI-geo.CA_CB_CG1_angle)),0,
    //                     sin(-(M_PI-geo.CA_CB_CG1_angle))), "CG2");
    return g;
}

cRigidGroup *makeIleGroup2(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG1");
    // g->addAtom(geo.R_CG1_CD1*cVector3(sin(geo.CB_CG1_CD1_angle-M_PI),
    //                     cos(geo.CB_CG1_CD1_angle-M_PI)*sin(geo.CB_CG1_CD1_angle),
    //                     cos(geo.CB_CG1_CD1_angle-M_PI)*cos(geo.CB_CG1_CD1_angle)), "CD1");
    g->addAtom(geo.R_CG1_CD1*cVector3(cos(-(M_PI-geo.CB_CG1_CD1_angle)),0,
                        sin(-(M_PI-geo.CB_CG1_CD1_angle))), "CD1");
    return g;
}

cRigidGroup *makeLeuGroup(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG1");
    g->addAtom(geo.R_CB_CG*cVector3(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.CG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.CG1_CB_CG2_angle/2.0)), "CD1");
    g->addAtom(geo.R_CB_CG*cVector3(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        -cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.CG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.CG1_CB_CG2_angle/2.0)), "CD2");
    return g;
}

cRigidGroup *makeThrGroup(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CB");
    g->addAtom(geo.R_CB_OG*cVector3(sin(geo.CA_CB_OG1_angle-M_PI/2.0),
                        cos(geo.CA_CB_OG1_angle-M_PI/2.0)*sin(geo.OG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_OG1_angle-M_PI/2.0)*cos(geo.OG1_CB_CG2_angle/2.0)), "OG1");
    g->addAtom(geo.R_CB_CG*cVector3(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        -cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.OG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.OG1_CB_CG2_angle/2.0)), "CG2");
    return g;
}

cRigidGroup *makeArgGroup(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CZ");
    g->addAtom(geo.R_CZ_NH*cVector3(sin(geo.NE_CZ_NH_angle-M_PI/2.0),
                        0,
                        cos(geo.NE_CZ_NH_angle-M_PI/2.0)), "NH1");
    g->addAtom(geo.R_CZ_NH*cVector3(sin(geo.NE_CZ_NH_angle-M_PI/2.0),
                        0,
                        -cos(geo.NE_CZ_NH_angle-M_PI/2.0)), "NH2");
    return g;
}

cRigidGroup *makeAspGroup(cGeometry &geo, std::string O1, std::string O2){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG");
    g->addAtom(geo.R_CG_OD*cVector3(sin(geo.CB_CG_OD_angle-M_PI/2.0),
                        0,
                        cos(geo.CB_CG_OD_angle-M_PI/2.0)), O1);
    g->addAtom(geo.R_CG_OD*cVector3(sin(geo.CB_CG_OD_angle-M_PI/2.0),
                        0,
                        -cos(geo.CB_CG_OD_angle-M_PI/2.0)), O2);
    return g;
}

cRigidGroup *makeAsnGroup(cGeometry &geo, std::string O1, std::string N2){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG");
    g->addAtom(geo.R_CG_OD1*cVector3(sin(geo.CB_CG_OD1_angle-M_PI/2.0),
                        0,
                        cos(geo.CB_CG_OD1_angle-M_PI/2.0)), O1);
    g->addAtom(geo.R_CG_ND2*cVector3(sin(geo.CB_CG_ND2_angle-M_PI/2.0),
                        0,
                        -cos(geo.CB_CG_ND2_angle-M_PI/2.0)), N2);
    return g;
}