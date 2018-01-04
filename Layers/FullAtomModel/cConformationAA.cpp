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
    cNode *nC, *nCA, *nN, *nCB, *nOG;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbOG_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB, *bbOG;
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
    bbCB_transform = new cTransform(&geo.N_C_CA_CB_diangle, &geo.C_CA_CB_angle, geo.R_CA_CB);
    this->groups.push_back(bbCB);
    this->transforms.push_back(bbCB_transform);
    nCB = addNode(nCA, groups.back(), transforms.back());

    bbOG = makeAtom("OG");
    bbOG_transform = new cTransform(params[2], &geo.CA_CB_OG_angle, geo.R_CB_OG);
    this->groups.push_back(bbOG);
    this->transforms.push_back(bbOG_transform);
    nOG = addNode(nCB, groups.back(), transforms.back());

    return nC;
}


cNode *cConformation::addCys(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB, *nSG;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbSG_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB, *bbSG;
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

    bbSG = makeAtom("SG");
    bbSG_transform = new cTransform(&zero_const, &geo.CA_CB_SG_angle, geo.R_CB_SG);
    this->groups.push_back(bbSG);
    this->transforms.push_back(bbSG_transform);
    nSG = addNode(nCB, groups.back(), transforms.back());

    return nC;
}

cNode *cConformation::addVal(cNode *parentC, std::vector<double*> params){
    cNode *nC, *nCA, *nN, *nCB, *nCG1, *nCG2;
    cTransform *bbN_transform, *bbCA_transform, *bbC_transform, *bbCB_transform, *bbCG1_transform, *bbCG2_transform;
    cRigidGroup *bbCA, *bbC, *bbN, *bbCB, *bbCG1, *bbCG2;
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

    bbCG1 = makeAtom("CG1");
    bbCG1_transform = new cTransform(&zero_const, &geo.CA_CB_CG1_angle, geo.R_CB_CG);
    this->groups.push_back(bbCG1);
    this->transforms.push_back(bbCG1_transform);
    nCG1 = addNode(nCB, groups.back(), transforms.back());

    bbCG2 = makeAtom("CG2");
    bbCG2_transform = new cTransform(&zero_const, &geo.CA_CB_CG2_angle, geo.R_CB_CG);
    this->groups.push_back(bbCG2);
    this->transforms.push_back(bbCG2_transform);
    nCG2 = addNode(nCB, groups.back(), transforms.back());

    return nC;
}
