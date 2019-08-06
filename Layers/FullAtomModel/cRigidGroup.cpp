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
