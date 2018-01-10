#include "cConformation.h"

cNode *cConformation::addGly(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform;
    cRigidGroup *bbCA, *bbC, *bbN;
    // geo.gly();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());

    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());
    return nC;
}

cNode *cConformation::addAla(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    bbCB = makeAtom("CB");
    bbCB_transform = new cTransform(&zero_const, &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());

    return nC;
}

cNode *cConformation::addSer(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    //TODO: Should be unified in a single rigid group
    bbCB = makeSerGroup(geo);
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());

    return nC;
}


cNode *cConformation::addCys(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    bbCB = makeCysGroup(geo);
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());

    return nC;
}

cNode *cConformation::addVal(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    //TODO: Should be unified in a single rigid group
    bbCB = makeValGroup(geo);
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());

    return nC;
}

cNode *cConformation::addIle(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB, *nCG1;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG1_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB, *bbCG1;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    bbCB = makeIleGroup1(geo);
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());

    bbCG1 = makeIleGroup2(geo);
    bbCG1_transform = new cTransform(params[3], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCG1);
    this->transforms.push_back(bbCG1_transform);
    nCG1 = addNode(nCB, groups.back(), transforms.back());

    return nC;
}

cNode *cConformation::addLeu(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB, *nCG1;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG1_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB, *bbCG1;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    bbCB = makeAtom("CB");
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());

    bbCG1 = makeLeuGroup(geo);
    bbCG1_transform = new cTransform(params[3], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCG1);
    this->transforms.push_back(bbCG1_transform);
    nCG1 = addNode(nCB, groups.back(), transforms.back());

    return nC;
}

cNode *cConformation::addThr(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    //TODO: Should be unified in a single rigid group
    bbCB = makeThrGroup(geo);
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());

    return nC;
}

cNode *cConformation::addArg(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB, *nCG, *nCD, *nNE, *nCZ;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform, *bbCD_transform, *bbNE_transform, *bbCZ_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB, *bbCG, *bbCD, *bbNE, *bbCZ;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    bbCB = makeAtom("CB");
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());

    bbCG = makeAtom("CG");
    bbCG_transform = new cTransform(params[3], &geo.C_C_C_angle, geo.R_C_C);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());

    bbCD = makeAtom("CD");
    bbCD_transform = new cTransform(params[4], &geo.C_C_C_angle, geo.R_C_C);
    this->groups.push_back(bbCD);
    this->transforms.push_back(bbCD_transform);
    nCD = addNode(nCG, groups.back(), transforms.back());

    bbNE = makeAtom("NE");
    bbNE_transform = new cTransform(params[5], &geo.CG_CD_NE_angle, geo.R_CD_NE);
    this->groups.push_back(bbNE);
    this->transforms.push_back(bbNE_transform);
    nNE = addNode(nCD, groups.back(), transforms.back());

    bbCZ = makeArgGroup(geo);
    bbCZ_transform = new cTransform(params[6], &geo.CD_NE_CZ_angle, geo.R_NE_CZ);
    this->groups.push_back(bbCZ);
    this->transforms.push_back(bbCZ_transform);
    nCZ = addNode(nNE, groups.back(), transforms.back());

    return nC;
}

cNode *cConformation::addLys(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB, *nCG, *nCD, *nCE, *nNZ;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform, *bbCD_transform, *bbCE_transform, *bbNZ_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB, *bbCG, *bbCD, *bbCE, *bbNZ;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    bbCB = makeAtom("CB");
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());

    bbCG = makeAtom("CG");
    bbCG_transform = new cTransform(params[3], &geo.C_C_C_angle, geo.R_C_C);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());

    bbCD = makeAtom("CD");
    bbCD_transform = new cTransform(params[4], &geo.C_C_C_angle, geo.R_C_C);
    this->groups.push_back(bbCD);
    this->transforms.push_back(bbCD_transform);
    nCD = addNode(nCG, groups.back(), transforms.back());

    bbCE = makeAtom("CE");
    bbCE_transform = new cTransform(params[5], &geo.C_C_C_angle, geo.R_CD_CE);
    this->groups.push_back(bbCE);
    this->transforms.push_back(bbCE_transform);
    nCE = addNode(nCD, groups.back(), transforms.back());

    bbNZ = makeAtom("NZ");
    bbNZ_transform = new cTransform(params[6], &geo.CD_CE_NZ_angle, geo.R_CE_NZ);
    this->groups.push_back(bbNZ);
    this->transforms.push_back(bbNZ_transform);
    nNZ = addNode(nCE, groups.back(), transforms.back());

    return nC;
}

cNode *cConformation::addAsp(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB, *nCG;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB, *bbCG;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    //TODO: Should be unified in a single rigid group
    bbCB = makeAtom("CB");
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());
    
    bbCG = makeAspGroup(geo, "OD1", "OD2");
    bbCG_transform = new cTransform(params[3], &geo.C_C_C_angle, geo.R_C_C);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());


    return nC;
}

cNode *cConformation::addAsn(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB, *nCG;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB, *bbCG;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    bbCB = makeAtom("CB");
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());
    
    bbCG = makeAsnGroup(geo, "OD1", "ND2");
    bbCG_transform = new cTransform(params[3], &geo.C_C_C_angle, geo.R_C_C);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());


    return nC;
}

cNode *cConformation::addGlu(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB, *nCG, *nCD;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform, *bbCD_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB, *bbCG, *bbCD;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    bbCB = makeAtom("CB");
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());

    bbCG = makeAtom("CG");
    bbCG_transform = new cTransform(params[3], &geo.C_C_C_angle, geo.R_C_C);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());
    
    bbCD = makeAspGroup(geo, "OE1", "OE2");
    bbCD_transform = new cTransform(params[4], &geo.C_C_C_angle, geo.R_C_C);
    this->groups.push_back(bbCD);
    this->transforms.push_back(bbCD_transform);
    nCD = addNode(nCG, groups.back(), transforms.back());


    return nC;
}

cNode *cConformation::addGln(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB, *nCG, *nCD;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform, *bbCD_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB, *bbCG, *bbCD;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    bbCB = makeAtom("CB");
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());
    
    bbCG = makeAtom("CG");
    bbCG_transform = new cTransform(params[3], &geo.C_C_C_angle, geo.R_C_C);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());

    bbCD = makeAsnGroup(geo, "OE1", "NE2");
    bbCD_transform = new cTransform(params[4], &geo.C_C_C_angle, geo.R_C_C);
    this->groups.push_back(bbCD);
    this->transforms.push_back(bbCD_transform);
    nCD = addNode(nCG, groups.back(), transforms.back());


    return nC;
}

cNode *cConformation::addMet(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB, *nCG, *nSD, *nCE;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG_transform, *bbSD_transform, *bbCE_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB, *bbCG, *bbSD, *bbCE;
    // geo.ala();

    bbN = makeAtom("N");
    if(parentC==NULL)
        bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    else
        bbN_transform = new cTransform(&geo.omega_const, &geo.CA_C_N_angle, geo.R_C_N);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    nN = addNode(parentC, groups.back(), transforms.back());

    bbCA = makeAtom("CA");
    bbCA_transform = new cTransform(params[0], &geo.C_N_CA_angle, geo.R_N_CA);
    groups.push_back(bbCA);
    transforms.push_back(bbCA_transform);
    nCA = addNode(nN, groups.back(), transforms.back());
    
    bbC = makeCarbonyl(geo);
    bbC_transform = new cTransform(params[1], &geo.N_CA_C_angle, geo.R_CA_C);
    this->groups.push_back(bbC);
    this->transforms.push_back(bbC_transform);
    nC = addNode(nCA, groups.back(), transforms.back());

    bbCB = makeAtom("CB");
    bbCB_transform = new cTransform(params[2], &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());
    
    bbCG = makeAtom("CG");
    bbCG_transform = new cTransform(params[3], &geo.C_C_C_angle, geo.R_C_C);
    this->groups.push_back(bbCG);
    this->transforms.push_back(bbCG_transform);
    nCG = addNode(nCB, groups.back(), transforms.back());

    bbSD = makeAtom("SD");
    bbSD_transform = new cTransform(params[4], &geo.CB_CG_SD_angle, geo.R_CG_SD);
    this->groups.push_back(bbSD);
    this->transforms.push_back(bbSD_transform);
    nSD = addNode(nCG, groups.back(), transforms.back());

    bbCE = makeAtom("CE");
    bbCE_transform = new cTransform(params[5], &geo.CG_SD_CE_angle, geo.R_SD_CE);
    this->groups.push_back(bbCE);
    this->transforms.push_back(bbCE_transform);
    nCE = addNode(nSD, groups.back(), transforms.back());


    return nC;
}