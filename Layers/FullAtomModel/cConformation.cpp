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
            std::cout<<"cTransform::getT Axis selection error"<<std::endl;
            break;
    };
    cMatrix44 m;
    m.ones();
    m.m[ax_ind][3]=dist;
    return m;
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
    mat = getRy(*beta)*getRx(*alpha)*getT(d, 'x');
}
void cTransform::print(){
    mat.print();
}


std::ostream& operator<<(std::ostream& os, const cNode& node){
    return os<<*(node.group);
}

cConformation::cConformation(std::string aa, double *alpha, double *beta){
    zero_const = 0.0;
    omega_const = OMEGACIS;
    kappa1 = KAPPA1;
    kappa2 = KAPPA2;
    kappa3 = KAPPA3;
    
    cRigidGroup *bbN = makeAtom("N");
    cTransform *bbN_transform = new cTransform(&zero_const, &zero_const, zero_const);
    groups.push_back(bbN);
    transforms.push_back(bbN_transform);
    cNode *nC, *nCa, *nN;
    nN = addNode(NULL, groups.back(), transforms.back());

    for(int i=0; i<aa.length(); i++){
        
        cRigidGroup *bbCa = makeAtom("CA");
        cTransform *bbCa_transform = new cTransform(alpha+i, &kappa1, R_N_CALPHA);
        groups.push_back(bbCa);
        transforms.push_back(bbCa_transform);
        nCa = addNode(nN, groups.back(), transforms.back());

        cRigidGroup *bbC = makeAtom("C");
        cTransform *bbC_transform = new cTransform(beta+i, &kappa2, R_CALPHA_C);
        this->groups.push_back(bbC);
        this->transforms.push_back(bbC_transform);
        nC = addNode(nCa, groups.back(), transforms.back());

        bbN = makeAtom("N");
        bbN_transform = new cTransform(&omega_const, &kappa3, R_C_N);
        this->groups.push_back(bbN);
        this->transforms.push_back(bbN_transform);
        nN = addNode(nC, groups.back(), transforms.back());
    }
}
cConformation::~cConformation(){
    delete [] nodes;
    delete [] transforms;
    delete [] groups;
}

cNode *cConformation::addNode(cNode *parent, cRigidGroup *group, cTransform *t){
    cNode *new_node = new cNode();
    new_node->group = group;
    new_node->T = t;
    new_node->parent = parent;
    nodes.push_back(new_node);
    if(parent==NULL){
        root = new_node;
    }else if(parent->left == NULL){
        parent->left = new_node;
    }else if(parent->right == NULL){
        parent->right = new_node;
    }else{
        std::cout<<"cConformation::addNode too many children"<<std::endl;
    }
    return new_node;
}

void cConformation::update(cNode *node){
    if(node->parent!=NULL){
        node->T->updateMatrix();
        node->M = (node->parent->M) * (node->T->mat);
        node->group->applyTransform(node->M);
    }else{
        node->T->updateMatrix();
        node->M.ones();
        node->group->applyTransform(node->M);
    }
    if(node->left!=NULL){
        update(node->left);
    }
    if(node->right!=NULL){
        update(node->right);
    }
}

void cConformation::print(cNode *node){
    std::cout<<*node;
    if(node->left!=NULL){
        std::cout<<"---";
        print(node->left);
    }else if(node->right!=NULL){
        std::cout<<"|\n";
        std::cout<<"---";
        print(node->right);
    }else{
        std::cout<<".\n";
    }
}
