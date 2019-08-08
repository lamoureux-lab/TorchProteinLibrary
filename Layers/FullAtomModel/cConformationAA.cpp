#include "cConformation.h"

#define PARENT_CHECK \
    if(parentC!=NULL){ \
        residueIndex = parentC->group->residueIndex + 1; \
        firstAtomIndex = parentC->group->atomIndexes.back() + 1; \
    }else{ \
        residueIndex = 0; \
        firstAtomIndex = 0; \
    } 

#define ADD_NITROGEN \
    bbN = makeAtom("N", firstAtomIndex, residueName, residueIndex, atoms_global); \
    if(parentC==NULL) \
        bbN_transform = new cTransform<T>(&zero_const, &zero_const, zero_const, NULL); \
    else \
        bbN_transform = new cTransform<T>(params[2], &geo.CA_C_N_angle, geo.R_C_N, params_grad[2]); \
    this->groups.push_back(bbN); \
    this->transforms.push_back(bbN_transform); \
    nN = addNode(parentC, groups.back(), transforms.back());

#define ADD_CARBON_ALPHA \
    bbCA = makeAtom("CA", firstAtomIndex + 1, residueName, residueIndex, atoms_global); \
    bbCA_transform = new cTransform<T>(params[0], &geo.C_N_CA_angle, geo.R_N_CA, params_grad[0]); \
    groups.push_back(bbCA); \
    transforms.push_back(bbCA_transform); \
    nCA = addNode(nN, groups.back(), transforms.back()); 

#define ADD_DUMMY_TRANSFORM \
    cTransform<T> *dummy_transform = new cTransform<T>(&geo.N_C_CA_CB_diangle, &geo.correction_angle, 0.0, NULL); \
    cRigidGroup<T> *dummy_group = new cRigidGroup<T>(); \
    this->groups.push_back(dummy_group); \
    this->transforms.push_back(dummy_transform); \
    cNode<T> *dummy_node = addNode(nCA, groups.back(), transforms.back()); 

#define ADD_CARBON_BETA(x, y) \
    bbCB = makeAtom("CB", firstAtomIndex+2, residueName, residueIndex, atoms_global); \
    bbCB_transform = new cTransform<T>(x, &geo.C_CA_CB_angle, geo.R_CA_CB, y); \
    this->groups.push_back(bbCB); \
    this->transforms.push_back(bbCB_transform); \
    nCB = addNode(dummy_node, groups.back(), transforms.back());

#define ADD_CARBON_GAMMA \
    bbCG = makeAtom("CG", firstAtomIndex+3, residueName, residueIndex, atoms_global); \
    bbCG_transform = new cTransform<T>(params[4], &geo.C_C_C_angle, geo.R_C_C, params_grad[4]); \
    this->groups.push_back(bbCG); \
    this->transforms.push_back(bbCG_transform); \
    nCG = addNode(nCB, groups.back(), transforms.back());

#define ADD_CARBON_DELTA \
    bbCD = makeAtom("CD", firstAtomIndex+4, residueName, residueIndex, atoms_global); \
    bbCD_transform = new cTransform<T>(params[5], &geo.C_C_C_angle, geo.R_C_C, params_grad[5]); \
    this->groups.push_back(bbCD); \
    this->transforms.push_back(bbCD_transform); \
    nCD = addNode(nCG, groups.back(), transforms.back());

#define ADD_CARBONYL(x) \
    bbC = makeCarbonyl(geo, firstAtomIndex + x, residueName, residueIndex, atoms_global, terminal); \
    bbC_transform = new cTransform<T>(params[1], &geo.N_CA_C_angle, geo.R_CA_C, params_grad[1]); \
    this->groups.push_back(bbC); \
    this->transforms.push_back(bbC_transform); \
    nC = addNode(nCA, groups.back(), transforms.back());


template <typename T> cNode<T> *cConformation<T>::addGly(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'G';
    PARENT_CHECK
    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_CARBONYL(2)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addAla(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB;

    uint residueIndex, firstAtomIndex;
    char residueName = 'A';
    PARENT_CHECK
    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(&zero_const, NULL)
    ADD_CARBONYL(3)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addSer(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'S';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM

    bbCB = makeSerGroup(geo, firstAtomIndex+2, residueName, residueIndex, atoms_global);
    bbCB_transform = new cTransform<T>(params[3], &geo.C_CA_CB_angle, geo.R_CA_CB, params_grad[3]);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(dummy_node, groups.back(), transforms.back());

    ADD_CARBONYL(4)
    return nC;
}


template <typename T> cNode<T> *cConformation<T>::addCys(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'C';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM

    bbCB = makeCysGroup(geo, firstAtomIndex+2, residueName, residueIndex, atoms_global);
    bbCB_transform = new cTransform<T>(params[3], &geo.C_CA_CB_angle, geo.R_CA_CB, params_grad[3]);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(dummy_node, groups.back(), transforms.back());

    ADD_CARBONYL(4)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addVal(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'V';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM

    bbCB = makeValGroup(geo, firstAtomIndex+2, residueName, residueIndex, atoms_global);
    bbCB_transform = new cTransform<T>(params[3], &geo.C_CA_CB_angle, geo.R_CA_CB, params_grad[3]);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(dummy_node, groups.back(), transforms.back());

    ADD_CARBONYL(5)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addIle(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB, *nCG1;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG1_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG1;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'I';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM

    bbCB = makeIleGroup1(geo, firstAtomIndex+2, residueName, residueIndex, atoms_global);
    bbCB_transform = new cTransform<T>(params[3], &geo.C_CA_CB_angle, geo.R_CA_CB, params_grad[3]);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(dummy_node, groups.back(), transforms.back());

    bbCG1 = makeIleGroup2(geo, firstAtomIndex+4, residueName, residueIndex, atoms_global);
    bbCG1_transform = new cTransform<T>(params[4], &geo.C_CA_CB_angle, geo.R_CA_CB, params_grad[4]);
    this->groups.push_back(bbCG1);
    this->transforms.push_back(bbCG1_transform);
    nCG1 = addNode(nCB, groups.back(), transforms.back());

    ADD_CARBONYL(6)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addLeu(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB, *nCG1;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG1_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG1;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'L';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(params[3], params_grad[3])

    bbCG1 = makeLeuGroup(geo, firstAtomIndex+3, residueName, residueIndex, atoms_global);
    bbCG1_transform = new cTransform<T>(params[4], &geo.C_CA_CB_angle, geo.R_CA_CB, params_grad[4]);
    this->groups.push_back(bbCG1);
    this->transforms.push_back(bbCG1_transform);
    nCG1 = addNode(nCB, groups.back(), transforms.back());

    ADD_CARBONYL(6)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addThr(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'T';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM

    bbCB = makeThrGroup(geo, firstAtomIndex+2, residueName, residueIndex, atoms_global);
    bbCB_transform = new cTransform<T>(params[3], &geo.C_CA_CB_angle, geo.R_CA_CB, params_grad[3]);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(dummy_node, groups.back(), transforms.back());

    ADD_CARBONYL(5)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addArg(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB, *nCG, *nCD, *nNE, *nCZ;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform, *bbCD_transform, *bbNE_transform, *bbCZ_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG, *bbCD, *bbNE, *bbCZ;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'R';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(params[3], params_grad[3])
    ADD_CARBON_GAMMA
    ADD_CARBON_DELTA

    bbNE = makeAtom("NE", firstAtomIndex+5, residueName, residueIndex, atoms_global);
    bbNE_transform = new cTransform<T>(params[6], &geo.CG_CD_NE_angle, geo.R_CD_NE, params_grad[6]);
    this->groups.push_back(bbNE);
    this->transforms.push_back(bbNE_transform);
    nNE = addNode(nCD, groups.back(), transforms.back());

    bbCZ = makeArgGroup(geo, firstAtomIndex+6, residueName, residueIndex, atoms_global);
    bbCZ_transform = new cTransform<T>(params[7], &geo.CD_NE_CZ_angle, geo.R_NE_CZ, params_grad[7]);
    this->groups.push_back(bbCZ);
    this->transforms.push_back(bbCZ_transform);
    nCZ = addNode(nNE, groups.back(), transforms.back());

    ADD_CARBONYL(9)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addLys(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB, *nCG, *nCD, *nCE, *nNZ;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform, *bbCD_transform, *bbCE_transform, *bbNZ_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG, *bbCD, *bbCE, *bbNZ;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'K';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(params[3], params_grad[3])
    ADD_CARBON_GAMMA
    ADD_CARBON_DELTA

    bbCE = makeAtom("CE", firstAtomIndex+5, residueName, residueIndex, atoms_global);
    bbCE_transform = new cTransform<T>(params[6], &geo.C_C_C_angle, geo.R_CD_CE, params_grad[6]);
    this->groups.push_back(bbCE);
    this->transforms.push_back(bbCE_transform);
    nCE = addNode(nCD, groups.back(), transforms.back());

    bbNZ = makeAtom("NZ", firstAtomIndex+6, residueName, residueIndex, atoms_global);
    bbNZ_transform = new cTransform<T>(params[7], &geo.CD_CE_NZ_angle, geo.R_CE_NZ, params_grad[7]);
    this->groups.push_back(bbNZ);
    this->transforms.push_back(bbNZ_transform);
    nNZ = addNode(nCE, groups.back(), transforms.back());

    ADD_CARBONYL(7)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addAsp(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB, *nCG;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'D';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(params[3], params_grad[3])
    
    bbCG = makeAspGroup(geo, "CG", "OD1", "OD2", firstAtomIndex+3, residueName, residueIndex, atoms_global);
    bbCG_transform = new cTransform<T>(params[4], &geo.C_C_C_angle, geo.R_C_C, params_grad[4]);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());

    ADD_CARBONYL(6)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addAsn(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB, *nCG;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'N';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(params[3], params_grad[3])
    
    bbCG = makeAsnGroup(geo, "CG", "OD1", "ND2", firstAtomIndex+3, residueName, residueIndex, atoms_global);
    bbCG_transform = new cTransform<T>(params[4], &geo.C_C_C_angle, geo.R_C_C, params_grad[4]);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());

    ADD_CARBONYL(6)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addGlu(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB, *nCG, *nCD;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform, *bbCD_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG, *bbCD;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'E';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(params[3], params_grad[3])
    ADD_CARBON_GAMMA
    
    bbCD = makeAspGroup(geo, "CD", "OE1", "OE2", firstAtomIndex+4, residueName, residueIndex, atoms_global);
    bbCD_transform = new cTransform<T>(params[5], &geo.C_C_C_angle, geo.R_C_C, params_grad[5]);
    this->groups.push_back(bbCD);
    this->transforms.push_back(bbCD_transform);
    nCD = addNode(nCG, groups.back(), transforms.back());

    ADD_CARBONYL(7)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addGln(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB, *nCG, *nCD;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform, *bbCD_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG, *bbCD;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'Q';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(params[3], params_grad[3])
    ADD_CARBON_GAMMA

    bbCD = makeAsnGroup(geo, "CD", "OE1", "NE2", firstAtomIndex+4, residueName, residueIndex, atoms_global);
    bbCD_transform = new cTransform<T>(params[5], &geo.C_C_C_angle, geo.R_C_C, params_grad[5]);
    this->groups.push_back(bbCD);
    this->transforms.push_back(bbCD_transform);
    nCD = addNode(nCG, groups.back(), transforms.back());

    ADD_CARBONYL(7)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addMet(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB, *nCG, *nSD, *nCE;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform, *bbSD_transform, *bbCE_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG, *bbSD, *bbCE;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'M';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(params[3], params_grad[3])
    ADD_CARBON_GAMMA

    bbSD = makeAtom("SD", firstAtomIndex+4, residueName, residueIndex, atoms_global);
    bbSD_transform = new cTransform<T>(params[5], &geo.CB_CG_SD_angle, geo.R_CG_SD, params_grad[5]);
    this->groups.push_back(bbSD);
    this->transforms.push_back(bbSD_transform);
    nSD = addNode(nCG, groups.back(), transforms.back());

    bbCE = makeAtom("CE", firstAtomIndex+5, residueName, residueIndex, atoms_global);
    bbCE_transform = new cTransform<T>(params[6], &geo.CG_SD_CE_angle, geo.R_SD_CE, params_grad[6]);
    this->groups.push_back(bbCE);
    this->transforms.push_back(bbCE_transform);
    nCE = addNode(nSD, groups.back(), transforms.back());

    ADD_CARBONYL(6)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addHis(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN, *nCB, *nCG;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'H';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(params[3], params_grad[3])
    
    bbCG = makeHisGroup(geo, firstAtomIndex+3, residueName, residueIndex, atoms_global);
    bbCG_transform = new cTransform<T>(params[4], &geo.C_C_C_angle, geo.R_C_C, params_grad[4]);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());

    ADD_CARBONYL(8)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addPro(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
    cNode<T> *nC, *nCA, *nN;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'P';
    PARENT_CHECK
    ADD_NITROGEN

    bbCA = makeProGroup(geo, firstAtomIndex+1, residueName, residueIndex, atoms_global);
    bbCA_transform = new cTransform<T>(params[0], &geo.C_N_CA_angle, geo.R_N_CA, params_grad[0]);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    ADD_CARBONYL(5)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addPhe(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
   cNode<T> *nC, *nCA, *nN, *nCB, *nCG;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG;

    uint residueIndex, firstAtomIndex;
    char residueName = 'F';
    PARENT_CHECK
    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(params[3], params_grad[3])
    
    bbCG = makePheGroup(geo, firstAtomIndex+3, residueName, residueIndex, atoms_global);
    bbCG_transform = new cTransform<T>(params[4], &geo.C_C_C_angle, geo.R_C_C, params_grad[4]);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());

    ADD_CARBONYL(9)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addTyr(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
   cNode<T> *nC, *nCA, *nN, *nCB, *nCG;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG;
    
    uint residueIndex, firstAtomIndex;
    char residueName = 'Y';
    PARENT_CHECK

    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(params[3], params_grad[3])
    
    bbCG = makeTyrGroup(geo, firstAtomIndex+3, residueName, residueIndex, atoms_global);
    bbCG_transform = new cTransform<T>(params[4], &geo.C_C_C_angle, geo.R_C_C, params_grad[4]);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());

    ADD_CARBONYL(10)
    return nC;
}

template <typename T> cNode<T> *cConformation<T>::addTrp(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal){
   cNode<T> *nC, *nCA, *nN, *nCB, *nCG;
    cTransform<T> *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform;
    cRigidGroup<T> *bbCA, *bbC, *bbN, *bbCB, *bbCG;

    uint residueIndex, firstAtomIndex;
    char residueName = 'W';
    PARENT_CHECK
    ADD_NITROGEN
    ADD_CARBON_ALPHA
    ADD_DUMMY_TRANSFORM
    ADD_CARBON_BETA(params[3], params_grad[3])
    
    bbCG = makeTrpGroup(geo, firstAtomIndex+3, residueName, residueIndex, atoms_global);
    bbCG_transform = new cTransform<T>(params[4], &geo.C_C_C_angle, geo.R_C_C, params_grad[4]);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());

    ADD_CARBONYL(12)
    return nC;
}

template class cConformation<float>;
template class cConformation<double>;