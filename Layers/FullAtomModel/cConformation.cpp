#include "cConformation.h"

#define KAPPA1 (3.14159 - 1.9391)
#define KAPPA2 (3.14159 - 2.061)
#define KAPPA3 (3.14159 -2.1186)
#define OMEGACIS -3.1318
#define R_CALPHA_C 1.525
#define R_C_N 1.330
#define R_N_CALPHA 1.460

cMatrix44 cTransform::getT(double dist, char axis){
    int ax_ind;
    switch(axis){
        case 'x':
            ax_ind=0;
            break;
        case 'y':
            ax_ind=1;
            break;
        case 'z':
            ax_ind=2;
            break;
        default:
            std::cout<<'cTransform::getT Axis selection error'<<std::endl;
            break;
    };
    cMatrix44 m;
    m.ones();
    m[3][ax_ind]=dist;
}
cMatrix44 cTransform::getRz(double angle){
    cMatrix44 m;
    m.m[0][0]=cos(angle); 	m.m[0][1]=-sin(angle);	m.m[0][2]=0;	m.m[0][3]=0;
	m.m[1][0]=sin(angle);	m.m[1][1]=cos(angle);	m.m[1][2]=0;	m.m[1][3]=0;
	m.m[2][0]=0;            m.m[2][1]=0;	        m.m[2][2]=1;    m.m[2][3]=0;
	m.m[3][0]=0;			m.m[3][1]=0;			m.m[3][2]=0;	m.m[3][3]=1;
    return m;
}
cMatrix44 cTransform::getRy(double angle){
    cMatrix44 m;
    m.m[0][0]=cos(angle); 	m.m[0][1]=0;	m.m[0][2]=sin(angle);	m.m[0][3]=0;
	m.m[1][0]=0;	        m.m[1][1]=1;	m.m[1][2]=0;	        m.m[1][3]=0;
	m.m[2][0]=-sin(angle);  m.m[2][1]=0;    m.m[2][2]=cos(angle);   m.m[2][3]=0;
	m.m[3][0]=0;			m.m[3][1]=0;	m.m[3][2]=0;	        m.m[3][3]=1;
    return m;
}
cMatrix44 cTransform::getRx(double angle){
    cMatrix44 m;
    m.m[0][0]=1;    m.m[0][1]=0;	        m.m[0][2]=0;            m.m[0][3]=0;
	m.m[1][0]=0;	m.m[1][1]=cos(angle);	m.m[1][2]=-sin(angle);  m.m[1][3]=0;
	m.m[2][0]=0;    m.m[2][1]=sin(angle);   m.m[2][2]=cos(angle);   m.m[2][3]=0;
	m.m[3][0]=0;	m.m[3][1]=0;	        m.m[3][2]=0;            m.m[3][3]=1;
    return m;
}
void cTransform::updateMatrix(){
    mat = getRy(beta)*getRx(alpha)*getT(d);
}


cConformation::cConformation(std::string aa, double *alpha, double *beta){
    zero_const = 0.0;
    omega_const = OMEGACIS;
    kappa1 = KAPPA1;
    kappa2 = KAPPA2;
    kappa3 = KAPPA3;

    cRigidGroup bbN = makeAtom();
    cTransform bbN_transform = cTransform(&zero_const, &zero_const, zero_const);
    this->groups.push_back(bbN);
    this->transforms.push_back(bbN_transform);
    cNode *nC, *nCa, *nN;
    nN = addNode(NULL, &bbN, &bbN_transform);
    for(int i=0; i<aa.length(); i++){
        
        cRigidGroup bbCa = makeAtom();
        cTransform bbCa_transform = cTransform(alpha+i, &kappa1, R_N_CALPHA);
        this->transforms.push_back(bbCa);
        this->transforms.push_back(bbCa_transform);
        nCa = addNode(nN, &bbCa, &bbCa_transform);

        cRigidGroup bbC = makeAtom();
        cTransform bbC_transform = cTransform(beta+i, &kappa2, R_CALPHA_C);
        this->transforms.push_back(bbC);
        this->transforms.push_back(bbC_transform);
        nC = addNode(nCa, &bbC, &bbC_transform);

        cRigidGroup bbN = makeAtom();
        cTransform bbN_transform = cTransform(&omega_const, &kappa3, R_C_N);
        this->groups.push_back(bbN);
        this->transforms.push_back(bbN_transform);
        nN = addNode(nC, &bbN, &bbN_transform);
    }
}
cNode *cConformation::addNode(cNode *parent, cRigidGroup *group, cTransform *t){
    cNode new_node = cNode();
    new_node.group = group;
    new_node.T = t;
    new_node.parent = parent;
    this->nodes.push_back(new_node);
    if(parent==NULL){
        root = &new_node;
    }else if(parent->left == NULL){
        parent->left = &new_node;
    }else if(parent->right == NULL){
        parent->right = &new_node;
    }else{
        std::cout<<'cConformation::addNode too many children'<<std::endl;
    }
    return &new_node;
}

void cConformation::update(cNode *node){
    if(node->parent!=NULL){
        node->updateMatrix();
        node->M = node->T->mat * node->parent->M;
        node->group.applyTransform(node->M);
    }else{
        node->M.ones();
        node->group.applyTransform(node->M);
    }
    if(node->left!=NULL){
        update(node->left);
    }
    if(node->right!=NULL){
        update(node->right);
    }
}