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
