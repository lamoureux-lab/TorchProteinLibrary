#include <cRigidGroup.h>
template <typename T> cRigidGroup<T>::cRigidGroup(){
}

template <typename T> cRigidGroup<T>::~cRigidGroup(){
}

template <typename T> void cRigidGroup<T>::addAtom(cVector3<T> pos, std::string atomName, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    this->atoms_local.push_back(pos);
    this->atoms_global.push_back(cVector3<T>(atoms_global_ptr + 3*atomIndex));
    this->atomNames.push_back(atomName);
    this->atomIndexes.push_back(atomIndex);
    this->residueName = residueName;
    this->residueIndex = residueIndex;
}

template <typename T> void cRigidGroup<T>::applyTransform(cMatrix44<T> mat){
    for(int i=0; i<this->atoms_local.size(); i++){
        atoms_global[i] = mat*atoms_local[i];
    }
}

template <typename T> std::ostream& operator<<(std::ostream& os, const cRigidGroup<T>& rg){
    for(int i=0; i<rg.atomNames.size(); i++){
        os<<rg.atomNames[i]<<" "<<rg.atoms_global[i]<<"\n";
    }
    return os;
}

template <typename T> cRigidGroup<T> *makeAtom(std::string atomName, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), atomName, atomIndex, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeCarbonyl(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr, bool terminal){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "C", atomIndex, residueName, residueIndex, atoms_global_ptr);   
    g->addAtom(geo.R_CB_OG*cVector3<T>(
        cos(geo.CA_C_O_angle),
        0.0,
        sin(geo.CA_C_O_angle)), "O", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    if(terminal){
         g->addAtom(geo.R_CB_OG*cVector3<T>(
            cos(geo.CA_C_O_angle),
            0.0,
            -sin(geo.CA_C_O_angle)), "OXT", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    }
    
    return g;
}
template <typename T> cRigidGroup<T> *makeSerGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CB", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_OG*cVector3<T>(cos(-(M_PI-geo.CA_CB_OG_angle)),0,
                        sin(-(M_PI-geo.CA_CB_OG_angle))), "OG", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    return g;
}
template <typename T> cRigidGroup<T> *makeCysGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CB", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_SG*cVector3<T>(cos(-(M_PI-geo.CA_CB_SG_angle)),0,
                        sin(-(M_PI-geo.CA_CB_SG_angle))), "SG", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    return g;
}
template <typename T> cRigidGroup<T> *makeValGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CB", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_CG*cVector3<T>(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.CG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.CG1_CB_CG2_angle/2.0)), "CG1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_CG*cVector3<T>(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        -cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.CG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.CG1_CB_CG2_angle/2.0)), "CG2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeIleGroup1(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CB", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_CG*cVector3<T>(sin(geo.CA_CB_CG1_angle - M_PI/2.0),
                        cos(geo.CA_CB_CG1_angle - M_PI/2.0)*sin(geo.CG1_CB_CG2_angle),
                        cos(geo.CA_CB_CG1_angle - M_PI/2.0)*cos(geo.CG1_CB_CG2_angle)), "CG2", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeIleGroup2(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CG1", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG1_CD1*cVector3<T>(cos((M_PI-geo.CB_CG1_CD1_angle)),0,
                        -sin((M_PI-geo.CB_CG1_CD1_angle))), "CD1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeLeuGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CG", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_CG*cVector3<T>(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.CG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.CG1_CB_CG2_angle/2.0)), "CD1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_CG*cVector3<T>(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        -cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.CG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.CG1_CB_CG2_angle/2.0)), "CD2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeThrGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CB", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_OG*cVector3<T>(sin(geo.CA_CB_OG1_angle-M_PI/2.0),
                        cos(geo.CA_CB_OG1_angle-M_PI/2.0)*sin(geo.OG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_OG1_angle-M_PI/2.0)*cos(geo.OG1_CB_CG2_angle/2.0)), "OG1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CB_CG*cVector3<T>(sin(geo.CA_CB_CG1_angle-M_PI/2.0),
                        -cos(geo.CA_CB_CG1_angle-M_PI/2.0)*sin(geo.OG1_CB_CG2_angle/2.0),
                        cos(geo.CA_CB_CG1_angle-M_PI/2.0)*cos(geo.OG1_CB_CG2_angle/2.0)), "CG2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeArgGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CZ", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CZ_NH*cVector3<T>(sin(geo.NE_CZ_NH_angle-M_PI/2.0),
                        0,
                        cos(geo.NE_CZ_NH_angle-M_PI/2.0)), "NH1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CZ_NH*cVector3<T>(sin(geo.NE_CZ_NH_angle-M_PI/2.0),
                        0,
                        -cos(geo.NE_CZ_NH_angle-M_PI/2.0)), "NH2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeAspGroup(cGeometry<T> &geo, std::string C, std::string O1, std::string O2, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), C, atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_OD*cVector3<T>(sin(geo.CB_CG_OD_angle-M_PI/2.0),
                        0,
                        cos(geo.CB_CG_OD_angle-M_PI/2.0)), O1, atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_OD*cVector3<T>(sin(geo.CB_CG_OD_angle-M_PI/2.0),
                        0,
                        -cos(geo.CB_CG_OD_angle-M_PI/2.0)), O2, atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeAsnGroup(cGeometry<T> &geo, std::string C, std::string O1, std::string N2, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), C, atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_OD1*cVector3<T>(sin(geo.CB_CG_OD1_angle-M_PI/2.0),
                        0,
                        cos(geo.CB_CG_OD1_angle-M_PI/2.0)), O1, atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_ND2*cVector3<T>(sin(geo.CB_CG_ND2_angle-M_PI/2.0),
                        0,
                        -cos(geo.CB_CG_ND2_angle-M_PI/2.0)), N2, atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeHisGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CG", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_CD2*cVector3<T>(cos(geo.CB_CG_CD2_angle), sin(geo.CB_CG_CD2_angle), 0), "CD2", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_NE2*cVector3<T>(cos(geo.CB_CG_NE2_angle), sin(geo.CB_CG_NE2_angle), 0), "NE2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_CE1*cVector3<T>(cos(geo.CB_CG_CE1_angle), sin(geo.CB_CG_CE1_angle), 0), "CE1", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(geo.R_CG_ND1*cVector3<T>(cos(geo.CB_CG_ND1_angle), sin(geo.CB_CG_ND1_angle), 0), "ND1", atomIndex+4, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeProGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CA", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.36842372, -1.29540981, 0.70468246), "CB", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(-0.76137817, -2.21546205, 0.39299099), "CG", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(-1.97838492, -1.33927458, 0.35849352), "CD", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makePheGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CG", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.69229424, -1.20533343, 0), "CD1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.08228309, -1.20846014, 0), "CE1", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.77997771, -0.00625341, 0), "CZ", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.08768347,  1.19908002, 0), "CE2", atomIndex+4, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.69769462,  1.20220673, 0), "CD2", atomIndex+5, residueName, residueIndex, atoms_global_ptr);
    
    return g;
}
template <typename T> cRigidGroup<T> *makeTyrGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CG", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.69229424, -1.20533343, 0), "CD1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.08228309, -1.20846014, 0), "CE1", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.77997771, -0.00625341, 0), "CZ", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.08768347,  1.19908002, 0), "CE2", atomIndex+4, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.69769462,  1.20220673, 0), "CD2", atomIndex+5, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(4.16960749,  0.05377697, 0), "OH", atomIndex+6, residueName, residueIndex, atoms_global_ptr);
    
    return g;
}

template <typename T> cRigidGroup<T> *makeTrpGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "CG", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.829890578, -1.09003744, 0), "CD1", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.84949676,  1.15028973, 0), "CD2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.1363699,  -0.64570528, 0), "NE1", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.18136443,  0.71889842, 0), "CE2", atomIndex+4, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.61303306,  2.5301684, 0), "CE3", atomIndex+5, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(3.22086556,  1.65663275, 0), "CZ2", atomIndex+6, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(1.68979906,  3.42486545, 0), "CZ3", atomIndex+7, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.92849902,  3.0257584, 0), "CH2", atomIndex+8, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makePhosGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "P", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.56, -1.24, 0.64), "OP1", atomIndex+1, residueName, residueIndex, atoms_global_ptr); // was exp(-0.014, 1.375, -0.45) now from guassian -> OP1
    g->addAtom(cVector3<T>(0.32, 1.24, 0.79), "OP2", atomIndex+2, residueName, residueIndex, atoms_global_ptr); // was OP1(-0.17653, -1.23469, -0.805979) -> OP2 [original OP2 was (0.0135, -0.09, -1.5)] now from gauss
    return g;
}
//    g->addAtom(cVector3<T>(-0.184111, -1.287736, 0.84606), "OP1", atomIndex+1, residueName, residueIndex, atoms_global_ptr); // was exp(-0.014, 1.375, -0.45) now from guassian -> OP1
//    g->addAtom(cVector3<T>(1.40141, 0.67988, 0.365802), "OP2", atomIndex+2, residueName, residueIndex, atoms_global_ptr); // was OP1(-0.17653, -1.23469, -0.805979) -> OP2 [original OP2 was (0.0135, -0.09, -1.5)] now from gauss
template <typename T> cRigidGroup<T> *makeC2Group(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "C2'", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.56, -1.24, 0.64), "O2'", atomIndex+1, residueName, residueIndex, atoms_global_ptr);//(0, 1.413, 0) or (0, 0, 1.413)?
    return g;
}

template <typename T> cRigidGroup<T> *makeCytGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "N1", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.688, -1.216, 0), "C2", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.0405,  -2.2735, 0), "O2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.041,  -1.209, 0), "N3", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.7035,  -0.056, 0), "C4", atomIndex+4, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(4.0375,  -0.1026, 0), "N4", atomIndex+5, residueName, residueIndex, atoms_global_ptr); //changed for test, was (3.8475,  -0.056, 0) and previously was (3.8475,  0.632, 0)
    g->addAtom(cVector3<T>(2.027,  1.2, 0), "C5", atomIndex+6, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.688,  1.181, 0), "C6", atomIndex+7, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeThyGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "N1", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.675, -1.199, 0), "C2", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.0555,  -2.25, 0), "O2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.043,  -1.084, 0), "N3", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.783,  -0.073, 0), "C4", atomIndex+4, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(4.0063,  -0.03403, 0), "O4", atomIndex+5, residueName, residueIndex, atoms_global_ptr);  //changed for test, was (3.835,  -0.073, 0) and previously was (3.835,  -0.5595, 0)
    g->addAtom(cVector3<T>(2.009,  1.305, 0), "C5", atomIndex+6, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.7205,  2.6205, 0), "C7", atomIndex+7, residueName, residueIndex, atoms_global_ptr); //changed for test, was (2.0277205,  2.6205, 0)
    g->addAtom(cVector3<T>(0.674,  1.202, 0), "C6", atomIndex+8, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeAdeGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "N9", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.828, 1.095, 0), "C8", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.105,  0.793, 0), "N7", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.095,  -0.595, 0), "C5", atomIndex+3, residueName, residueIndex, atoms_global_ptr); // changed for test, was (1.95,  -0.2865, 0)
    g->addAtom(cVector3<T>(3.2,  -1.55, 0), "C6", atomIndex+4, residueName, residueIndex, atoms_global_ptr);// test, was (2.515,  -1.5735, 0)
    g->addAtom(cVector3<T>(4.455,  -1.25, 0), "N6", atomIndex+5, residueName, residueIndex, atoms_global_ptr);// test, was (3.68,  -2.2245, 0)
    g->addAtom(cVector3<T>(2.79,  -2.84, 0), "N1", atomIndex+6, residueName, residueIndex, atoms_global_ptr); // test, was (1.803,  -3.3725, 0)
    g->addAtom(cVector3<T>(1.5,  -3.2, 0), "C2", atomIndex+7, residueName, residueIndex, atoms_global_ptr);//test, was (0.798,  -3.5, 0)
    g->addAtom(cVector3<T>(0.397,  -2.427, 0), "N3", atomIndex+8, residueName, residueIndex, atoms_global_ptr);//test, was (0.227,  -2.298, 0)
    g->addAtom(cVector3<T>(0.829,  -1.096, 0), "C4", atomIndex+9, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeGuaGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "N9", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.824, 1.101, 0), "C8", atomIndex+1, residueName, residueIndex, atoms_global_ptr); //(0.824, 1.101, 0)
    g->addAtom(cVector3<T>(2.092,  0.792, 0), "N7", atomIndex+2, residueName, residueIndex, atoms_global_ptr); //(2.092,  0.792, 0)
    g->addAtom(cVector3<T>(2.092,  -0.596, 0), "C5", atomIndex+3, residueName, residueIndex, atoms_global_ptr); //(1.921,  -0.266, 0)
    g->addAtom(cVector3<T>(3.18,  -1.5, 0), "C6", atomIndex+4, residueName, residueIndex, atoms_global_ptr);// test, was (2.489,  -1.566, 0)
    g->addAtom(cVector3<T>(4.356,  -1.23, 0), "O6", atomIndex+5, residueName, residueIndex, atoms_global_ptr);// test, was (3.585,  -2.139, 0)
    g->addAtom(cVector3<T>(2.8,  -2.8, 0), "N1", atomIndex+6, residueName, residueIndex, atoms_global_ptr);// test, was (1.699,  -2.711, 0)
    g->addAtom(cVector3<T>(1.5,  -3.2, 0), "C2", atomIndex+7, residueName, residueIndex, atoms_global_ptr);// test, was (0.756333,  -3.399666, 0)
    g->addAtom(cVector3<T>(1.28,  -4.5, 0), "N2", atomIndex+8, residueName, residueIndex, atoms_global_ptr);// test, was (0.0651666,  -4.5481666, 0)
    g->addAtom(cVector3<T>(0.385,  -2.38, 0), "N3", atomIndex+9, residueName, residueIndex, atoms_global_ptr);//test, was (0.204,  -2.3, 0)
    g->addAtom(cVector3<T>(0.823,  -1.1, 0), "C4", atomIndex+10, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template <typename T> cRigidGroup<T> *makeUraGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr){
    cRigidGroup<T> *g = new cRigidGroup<T>();
    g->addAtom(cVector3<T>(0,0,0), "N1", atomIndex, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.688, -1.216, 0), "C2", atomIndex+1, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.0405,  -2.2735, 0), "O2", atomIndex+2, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.041,  -1.209, 0), "N3", atomIndex+3, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(2.7035,  -0.056, 0), "C4", atomIndex+4, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(4.0375,  -0.1026, 0), "O4", atomIndex+5, residueName, residueIndex, atoms_global_ptr); //Will need to be adjusted
    g->addAtom(cVector3<T>(2.027,  1.2, 0), "C5", atomIndex+6, residueName, residueIndex, atoms_global_ptr);
    g->addAtom(cVector3<T>(0.688,  1.181, 0), "C6", atomIndex+7, residueName, residueIndex, atoms_global_ptr);
    return g;
}

template class cRigidGroup<float>;
template cRigidGroup<float> *makeAtom(std::string, uint, char, uint, float*);
template cRigidGroup<float> *makeCarbonyl(cGeometry<float>&, uint, char, uint, float*, bool);
template cRigidGroup<float> *makeSerGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeCysGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeValGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeIleGroup1(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeIleGroup2(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeLeuGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeThrGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeArgGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeAspGroup(cGeometry<float>&, std::string, std::string, std::string, uint, char, uint, float*);
template cRigidGroup<float> *makeAsnGroup(cGeometry<float>&, std::string, std::string, std::string, uint, char, uint, float*);
template cRigidGroup<float> *makeHisGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeProGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makePheGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeTyrGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeTrpGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makePhosGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeC2Group(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeCytGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeThyGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeAdeGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeGuaGroup(cGeometry<float>&, uint, char, uint, float*);
template cRigidGroup<float> *makeUraGroup(cGeometry<float>&, uint, char, uint, float*);

template class cRigidGroup<double>;
template cRigidGroup<double> *makeAtom(std::string, uint, char, uint, double*);
template cRigidGroup<double> *makeCarbonyl(cGeometry<double>&, uint, char, uint, double*, bool);
template cRigidGroup<double> *makeSerGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeCysGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeValGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeIleGroup1(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeIleGroup2(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeLeuGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeThrGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeArgGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeAspGroup(cGeometry<double>&, std::string, std::string, std::string, uint, char, uint, double*);
template cRigidGroup<double> *makeAsnGroup(cGeometry<double>&, std::string, std::string, std::string, uint, char, uint, double*);
template cRigidGroup<double> *makeHisGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeProGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makePheGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeTyrGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeTrpGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makePhosGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeC2Group(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeCytGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeThyGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeAdeGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeGuaGroup(cGeometry<double>&, uint, char, uint, double*);
template cRigidGroup<double> *makeUraGroup(cGeometry<double>&, uint, char, uint, double*);
