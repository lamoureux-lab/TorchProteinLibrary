#include "cConformation.h"

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
cMatrix44 cTransform::getDRx(double angle){
    cMatrix44 m;
    m.m[0][0]=0;    m.m[0][1]=0;	        m.m[0][2]=0;            m.m[0][3]=0;
	m.m[1][0]=0;	m.m[1][1]=-sin(angle);	m.m[1][2]=-cos(angle);  m.m[1][3]=0;
	m.m[2][0]=0;    m.m[2][1]=cos(angle);   m.m[2][2]=-sin(angle);  m.m[2][3]=0;
	m.m[3][0]=0;	m.m[3][1]=0;	        m.m[3][2]=0;            m.m[3][3]=1;
    return m;
}
void cTransform::updateMatrix(){
    mat = getRy(*beta)*getRx(*alpha)*getT(d, 'x');
}
void cTransform::updateDMatrix(){
    dmat = getRy(*beta)*getDRx(*alpha)*getT(d, 'x');
}
void cTransform::print(){
    mat.print();
}


std::ostream& operator<<(std::ostream& os, const cNode& node){
    return os<<*(node.group);
}

cConformation::cConformation(std::string aa, double *data, int length){
    cNode *lastC = NULL;

    for(int i=0; i<aa.length(); i++){
        double *phi = data + i + length*0;
        double *psi = data + i + length*1;
        double *xi1 = data + i + length*2;
        double *xi2 = data + i + length*3;
        double *xi3 = data + i + length*4;
        double *xi4 = data + i + length*5;
        double *xi5 = data + i + length*6;
        std::vector<double*> params({phi, psi, xi1, xi2, xi3, xi4, xi5});
        switch(aa[i]){
            case 'G':
                lastC = addGly(lastC, params);
                break;
            case 'A':
                lastC = addAla(lastC, params);
                break;
            case 'S':
                lastC = addSer(lastC, params);
                break;
            case 'C':
                lastC = addCys(lastC, params);
                break;
            case 'V':
                lastC = addVal(lastC, params);
                break;
            case 'I':
                lastC = addIle(lastC, params);
                break;
            case 'L':
                lastC = addLeu(lastC, params);
                break;
            case 'T':
                lastC = addThr(lastC, params);
                break;
            case 'R':
                lastC = addArg(lastC, params);
                break;
            case 'K':
                lastC = addLys(lastC, params);
                break;
            case 'D':
                lastC = addAsp(lastC, params);
                break;
            case 'N':
                lastC = addAsn(lastC, params);
                break;
            case 'E':
                lastC = addGlu(lastC, params);
                break;
            case 'Q':
                lastC = addGln(lastC, params);
                break;
            case 'M':
                lastC = addMet(lastC, params);
                break;
            case 'H':
                lastC = addHis(lastC, params);
                break;
            case 'P':
                lastC = addPro(lastC, params);
                break;
            case 'F':
                lastC = addPhe(lastC, params);
                break;
            case 'Y':
                lastC = addTyr(lastC, params);
                break;
            case 'W':
                lastC = addTrp(lastC, params);
                break;
        }
    }
}
cConformation::~cConformation(){
    for(int i=0; i<nodes.size(); i++)
        delete nodes[i];
    for(int i=0; i<transforms.size(); i++)
        delete transforms[i];
    for(int i=0; i<groups.size(); i++)
        delete groups[i];
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
    }
    if(node->right!=NULL){
        std::cout<<"|\n";
        std::cout<<"---";
        print(node->right);
    }
    std::cout<<".\n";
    
}