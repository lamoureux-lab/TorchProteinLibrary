#include <cRigidGroup.h>

cRigidGroup::cRigidGroup(){
}

cRigidGroup::~cRigidGroup(){
    
}

void cRigidGroup::addAtom(cVector3 pos, std::string atomName){
    this->atoms_local.push_back(pos);
    this->atoms_global.push_back(pos);
    this->atomNames.push_back(atomName);
    this->atoms_grad.push_back(cVector3(0.0,0.0,0.0));
}

void cRigidGroup::applyTransform(cMatrix44 mat){
    for(int i=0; i<this->atoms_local.size(); i++){
        atoms_global[i] = mat*atoms_local[i];
    }
}

std::ostream& operator<<(std::ostream& os, const cRigidGroup& rg){
    for(int i=0; i<rg.atomNames.size(); i++){
        os<<rg.atomNames[i]<<" "<<rg.atoms_global[i]<<"\n";
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

cRigidGroup *makeHisGroup(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG");
    g->addAtom(geo.R_CG_CD2*cVector3(cos(geo.CB_CG_CD2_angle), sin(geo.CB_CG_CD2_angle), 0), "CD2");
    g->addAtom(geo.R_CG_NE2*cVector3(cos(geo.CB_CG_NE2_angle), sin(geo.CB_CG_NE2_angle), 0), "NE2");
    g->addAtom(geo.R_CG_CE1*cVector3(cos(geo.CB_CG_CE1_angle), sin(geo.CB_CG_CE1_angle), 0), "CE1");
    g->addAtom(geo.R_CG_ND1*cVector3(cos(geo.CB_CG_ND1_angle), sin(geo.CB_CG_ND1_angle), 0), "ND1");
    return g;
}

cRigidGroup *makeProGroup(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CA");
    g->addAtom(cVector3(0.36842372, -1.29540981, 0.70468246), "CB");
    g->addAtom(cVector3(-0.76137817, -2.21546205, 0.39299099), "CG");
    g->addAtom(cVector3(-1.97838492, -1.33927458, 0.35849352), "CD");
    return g;
}

cRigidGroup *makePheGroup(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG");
    g->addAtom(cVector3(0.69229424, -1.20533343, 0), "CD1");
    g->addAtom(cVector3(2.08228309, -1.20846014, 0), "CE1");
    g->addAtom(cVector3(2.77997771, -0.00625341, 0), "CZ");
    g->addAtom(cVector3(2.08768347,  1.19908002, 0), "CE2");
    g->addAtom(cVector3(0.69769462,  1.20220673, 0), "CD2");
    
    return g;
}
cRigidGroup *makeTyrGroup(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG");
    g->addAtom(cVector3(0.69229424, -1.20533343, 0), "CD1");
    g->addAtom(cVector3(2.08228309, -1.20846014, 0), "CE1");
    g->addAtom(cVector3(2.77997771, -0.00625341, 0), "CZ");
    g->addAtom(cVector3(2.08768347,  1.19908002, 0), "CE2");
    g->addAtom(cVector3(0.69769462,  1.20220673, 0), "CD2");
    g->addAtom(cVector3(4.16960749,  0.05377697, 0), "OH");
    
    return g;
}

cRigidGroup *makeTrpGroup(cGeometry &geo){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG");
    g->addAtom(cVector3(0.829890578, -1.09003744, 0), "CD1");
    g->addAtom(cVector3(0.84949676,  1.15028973, 0), "CD2");
    g->addAtom(cVector3(2.1363699,  -0.64570528, 0), "NE1");
    g->addAtom(cVector3(2.18136443,  0.71889842, 0), "CE2");
    g->addAtom(cVector3(0.61303306,  2.5301684, 0), "CE3");
    g->addAtom(cVector3(3.22086556,  1.65663275, 0), "CZ2");
    g->addAtom(cVector3(1.68979906,  3.42486545, 0), "CZ3");
    g->addAtom(cVector3(2.92849902,  3.0257584, 0), "CH2");
    return g;
}