#include <cRigidGroup.h>

cRigidGroup::cRigidGroup(){
}

cRigidGroup::~cRigidGroup(){
}

void cRigidGroup::addAtom(cVector3 pos, std::string atomName, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    this->atoms_local.push_back(pos);
    this->atoms_global.push_back(cVector3(atoms_global_ptr + 3*atomIndex));
    this->atomNames.push_back(atomName);
    this->atomIndexes.push_back(atomIndex);
    this->residueName = residueName;
    this->residueIndex = residueIndex;
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

cRigidGroup *makeAtom(std::string atomName, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), atomName, atomIndex, residueName, residueIndex, atoms_global_ptr);
    return g;
}

cRigidGroup *makeCarbonyl(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr, bool terminal){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "C", atomIndex, residueName, residueIndex, atoms_global_ptr);   
    g->addAtom(geo.R_CB_OG*cVector3(
        cos(geo.CA_C_O_angle),
        sin(geo.CA_C_O_angle)*cos(geo.CA_C_O_angle/2.0),
        sin(geo.CA_C_O_angle)*sin(geo.CA_C_O_angle/2.0)), "O", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    if(terminal){
        g->addAtom(geo.R_CB_OG*cVector3(
            cos(geo.CA_C_O_angle),
            sin(geo.CA_C_O_angle)*cos(geo.CA_C_O_angle/2.0),
            sin(geo.CA_C_O_angle)*sin(geo.CA_C_O_angle/2.0)), "OXT", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    }
    
    return g;
}
cRigidGroup *makeSerGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CB", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_OG*cVector3(cos(-(M_PI-geo.CA_CB_OG_angle)),0,
                        sin(-(M_PI-geo.CA_CB_OG_angle))), "OG", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    return g;
}
cRigidGroup *makeCysGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CB", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_SG*cVector3(cos(-(M_PI-geo.CA_CB_SG_angle)),0,
                        sin(-(M_PI-geo.CA_CB_SG_angle))), "SG", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    return g;
}
cRigidGroup *makeValGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CB", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_CG*cVector3(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.CG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.CG1_CB_CG2_angle/2.0)), "CG1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_CG*cVector3(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        -cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.CG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.CG1_CB_CG2_angle/2.0)), "CG2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    return g;
}

cRigidGroup *makeIleGroup1(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CB", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_CG*cVector3(sin(geo.CA_CB_CG1_angle - M_PI/2.0),
                        cos(geo.CA_CB_CG1_angle - M_PI/2.0)*sin(geo.CG1_CB_CG2_angle),
                        cos(geo.CA_CB_CG1_angle - M_PI/2.0)*cos(geo.CG1_CB_CG2_angle)), "CG2", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    return g;
}

cRigidGroup *makeIleGroup2(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG1", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG1_CD1*cVector3(cos((M_PI-geo.CB_CG1_CD1_angle)),0,
                        -sin((M_PI-geo.CB_CG1_CD1_angle))), "CD1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    return g;
}

cRigidGroup *makeLeuGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_CG*cVector3(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.CG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.CG1_CB_CG2_angle/2.0)), "CD1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_CG*cVector3(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        -cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.CG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.CG1_CB_CG2_angle/2.0)), "CD2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    return g;
}

cRigidGroup *makeThrGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CB", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_OG*cVector3(sin(geo.CA_CB_OG1_angle-M_PI/2.0),
                        cos(geo.CA_CB_OG1_angle-M_PI/2.0)*sin(geo.OG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_OG1_angle-M_PI/2.0)*cos(geo.OG1_CB_CG2_angle/2.0)), "OG1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_CG*cVector3(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        -cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.OG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.OG1_CB_CG2_angle/2.0)), "CG2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    return g;
}

cRigidGroup *makeArgGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CZ", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CZ_NH*cVector3(sin(geo.NE_CZ_NH_angle-M_PI/2.0),
                        0,
                        cos(geo.NE_CZ_NH_angle-M_PI/2.0)), "NH1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CZ_NH*cVector3(sin(geo.NE_CZ_NH_angle-M_PI/2.0),
                        0,
                        -cos(geo.NE_CZ_NH_angle-M_PI/2.0)), "NH2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    return g;
}

cRigidGroup *makeAspGroup(cGeometry &geo, std::string C, std::string O1, std::string O2, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), C, atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_OD*cVector3(sin(geo.CB_CG_OD_angle-M_PI/2.0),
                        0,
                        cos(geo.CB_CG_OD_angle-M_PI/2.0)), O1, atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_OD*cVector3(sin(geo.CB_CG_OD_angle-M_PI/2.0),
                        0,
                        -cos(geo.CB_CG_OD_angle-M_PI/2.0)), O2, atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    return g;
}

cRigidGroup *makeAsnGroup(cGeometry &geo, std::string C, std::string O1, std::string N2, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), C, atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_OD1*cVector3(sin(geo.CB_CG_OD1_angle-M_PI/2.0),
                        0,
                        cos(geo.CB_CG_OD1_angle-M_PI/2.0)), O1, atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_ND2*cVector3(sin(geo.CB_CG_ND2_angle-M_PI/2.0),
                        0,
                        -cos(geo.CB_CG_ND2_angle-M_PI/2.0)), N2, atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    return g;
}

cRigidGroup *makeHisGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_CD2*cVector3(cos(geo.CB_CG_CD2_angle), sin(geo.CB_CG_CD2_angle), 0), "CD2", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_NE2*cVector3(cos(geo.CB_CG_NE2_angle), sin(geo.CB_CG_NE2_angle), 0), "NE2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_CE1*cVector3(cos(geo.CB_CG_CE1_angle), sin(geo.CB_CG_CE1_angle), 0), "CE1", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_ND1*cVector3(cos(geo.CB_CG_ND1_angle), sin(geo.CB_CG_ND1_angle), 0), "ND1", atomIndex+4, residueName, residueIndex, atoms_global_ptr);
    return g;
}

cRigidGroup *makeProGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CA", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(0.36842372, -1.29540981, 0.70468246), "CB", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(-0.76137817, -2.21546205, 0.39299099), "CG", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(-1.97838492, -1.33927458, 0.35849352), "CD", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    return g;
}

cRigidGroup *makePheGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(0.69229424, -1.20533343, 0), "CD1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(2.08228309, -1.20846014, 0), "CE1", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(2.77997771, -0.00625341, 0), "CZ", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(2.08768347,  1.19908002, 0), "CE2", atomIndex+4, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(0.69769462,  1.20220673, 0), "CD2", atomIndex+5, residueName, residueIndex, atoms_global_ptr);
    
    return g;
}
cRigidGroup *makeTyrGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(0.69229424, -1.20533343, 0), "CD1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(2.08228309, -1.20846014, 0), "CE1", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(2.77997771, -0.00625341, 0), "CZ", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(2.08768347,  1.19908002, 0), "CE2", atomIndex+4, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(0.69769462,  1.20220673, 0), "CD2", atomIndex+5, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(4.16960749,  0.05377697, 0), "OH", atomIndex+6, residueName, residueIndex, atoms_global_ptr);
    
    return g;
}

cRigidGroup *makeTrpGroup(cGeometry &geo, uint atomIndex, char residueName, uint residueIndex, double *atoms_global_ptr){
    cRigidGroup *g = new cRigidGroup();
    g->addAtom(cVector3(0,0,0), "CG", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(0.829890578, -1.09003744, 0), "CD1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(0.84949676,  1.15028973, 0), "CD2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(2.1363699,  -0.64570528, 0), "NE1", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(2.18136443,  0.71889842, 0), "CE2", atomIndex+4, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(0.61303306,  2.5301684, 0), "CE3", atomIndex+5, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(3.22086556,  1.65663275, 0), "CZ2", atomIndex+6, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(1.68979906,  3.42486545, 0), "CZ3", atomIndex+7, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3(2.92849902,  3.0257584, 0), "CH2", atomIndex+8, residueName, residueIndex, atoms_global_ptr);
    return g;
}