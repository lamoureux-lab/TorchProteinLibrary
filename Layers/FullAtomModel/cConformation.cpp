#include "cConformation.h"

#include <nUtil.h>
using namespace StringUtil;
using namespace ProtUtil;

void cTransform::updateMatrix(){
    cMatrix44 Ry, T, Rx;
    Ry.setRy(*beta);
    T.setT(d, 'x');
    Rx.setRx(*alpha);
    mat = Ry*T*Rx;
}
void cTransform::updateDMatrix(){
    cMatrix44 Ry, T, DRx;
    Ry.setRy(*beta);
    T.setT(d, 'x');
    DRx.setDRx(*alpha);
    dmat = Ry*T*DRx;
}
void cTransform::print(){
    mat.print();
}


std::ostream& operator<<(std::ostream& os, const cNode& node){
    return os<<*(node.group);
}

cConformation::cConformation(std::string aa, double *angles, double *angles_grad, uint angles_length, double *atoms_global, bool add_terminal){
    cNode *lastC = NULL;
    zero_const = 0.0;
    this->atoms_global = atoms_global;
    bool terminal = false;
    for(int i=0; i<aa.length(); i++){
        double *phi = angles + i + angles_length*0;double *dphi = angles_grad + i + angles_length*0;
        double *psi = angles + i + angles_length*1;double *dpsi = angles_grad + i + angles_length*1;
        double *xi1 = angles + i + angles_length*2;double *dxi1 = angles_grad + i + angles_length*2;
        double *xi2 = angles + i + angles_length*3;double *dxi2 = angles_grad + i + angles_length*3;
        double *xi3 = angles + i + angles_length*4;double *dxi3 = angles_grad + i + angles_length*4;
        double *xi4 = angles + i + angles_length*5;double *dxi4 = angles_grad + i + angles_length*5;
        double *xi5 = angles + i + angles_length*6;double *dxi5 = angles_grad + i + angles_length*6;
        std::vector<double*> params({phi, psi, xi1, xi2, xi3, xi4, xi5});
        std::vector<double*> params_grad({dphi, dpsi, dxi1, dxi2, dxi3, dxi4, dxi5});
        if(add_terminal){
            if(i == (aa.length()-1))
                terminal = true;
            else
                terminal = false;
        }
        switch(aa[i]){
            case 'G':
                lastC = addGly(lastC, params, params_grad, terminal);
                break;
            case 'A':
                lastC = addAla(lastC, params, params_grad, terminal);
                break;
            case 'S':
                lastC = addSer(lastC, params, params_grad, terminal);
                break;
            case 'C':
                lastC = addCys(lastC, params, params_grad, terminal);
                break;
            case 'V':
                lastC = addVal(lastC, params, params_grad, terminal);
                break;
            case 'I':
                lastC = addIle(lastC, params, params_grad, terminal);
                break;
            case 'L':
                lastC = addLeu(lastC, params, params_grad, terminal);
                break;
            case 'T':
                lastC = addThr(lastC, params, params_grad, terminal);
                break;
            case 'R':
                lastC = addArg(lastC, params, params_grad, terminal);
                break;
            case 'K':
                lastC = addLys(lastC, params, params_grad, terminal);
                break;
            case 'D':
                lastC = addAsp(lastC, params, params_grad, terminal);
                break;
            case 'N':
                lastC = addAsn(lastC, params, params_grad, terminal);
                break;
            case 'E':
                lastC = addGlu(lastC, params, params_grad, terminal);
                break;
            case 'Q':
                lastC = addGln(lastC, params, params_grad, terminal);
                break;
            case 'M':
                lastC = addMet(lastC, params, params_grad, terminal);
                break;
            case 'H':
                lastC = addHis(lastC, params, params_grad, terminal);
                break;
            case 'P':
                lastC = addPro(lastC, params, params_grad, terminal);
                break;
            case 'F':
                lastC = addPhe(lastC, params, params_grad, terminal);
                break;
            case 'Y':
                lastC = addTyr(lastC, params, params_grad, terminal);
                break;
            case 'W':
                lastC = addTrp(lastC, params, params_grad, terminal);
                break;
        }
    }
    
    //Computing conformation
    this->update(this->root);
    //Computing number of atoms
    this->num_atoms = 0;
    for(int i=0; i<groups.size();i++){
        num_atoms += groups[i]->atoms_global.size();
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
       throw(std::string("cConformation::addNode too many children"));
    }
    return new_node;
}

void cConformation::update(cNode *node){
    node->T->updateMatrix();
    node->T->updateDMatrix();
    if(node->parent!=NULL){    
        node->M = (node->parent->M) * (node->T->mat);
        node->F = (node->parent->M) * (node->T->dmat) * invertTransform44(node->M);
        node->group->applyTransform(node->M);
    }else{
        node->M.setIdentity();
        node->F = (node->T->dmat) * invertTransform44(node->M);
        node->group->applyTransform(node->M);
    }
    if(node->left!=NULL){
        update(node->left);
    }
    if(node->right!=NULL){
        update(node->right);
    }
}

double cConformation::backward(cNode *root_node, cNode *node, double *atoms_grad){
    double grad = 0.0;
    if(node->left!=NULL){
        grad += backward(root_node, node->left, atoms_grad);
    }
    if(node->right!=NULL){
        grad += backward(root_node, node->right, atoms_grad);
    }
    for(int i=0;i<node->group->atoms_global.size(); i++){
        cVector3 gradVec(
            atoms_grad[node->group->atomIndexes[i]*3 + 0],
            atoms_grad[node->group->atomIndexes[i]*3 + 1],
            atoms_grad[node->group->atomIndexes[i]*3 + 2]
        );
        
        grad += gradVec | (root_node->F * node->group->atoms_global[i]);
        
        
    }
    /*if( fabs(grad)>10.0){
        grad = copysignf(1.0, grad)*10.0;
    }*/
    return grad;
}


void cConformation::backward(cNode *node, double *atoms_grad){
    for(int i=0; i<nodes.size();i++){
        if(nodes[i]->T->grad_alpha!=NULL) 
            *(nodes[i]->T->grad_alpha) = backward(nodes[i], nodes[i], atoms_grad);
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

void cConformation::save(std::string filename, const char mode){
    if(mode=='w'){
        std::ofstream pfile(filename, std::ofstream::out);
    
        for(int i=0; i<groups.size(); i++){
            for(int j=0; j<groups[i]->atoms_global.size(); j++){
                cVector3 r;
                r = groups[i]->atoms_global[j];
                std::string atom_name;
                atom_name = groups[i]->atomNames[j];
                int atom_index = groups[i]->atomIndexes[j];
                int res_index = groups[i]->residueIndex;
                std::string res_name = convertRes1to3(groups[i]->residueName);
                pfile<<string_format("ATOM  %5d %4s %3s A%4d    %8.3f%8.3f%8.3f\n", atom_index, atom_name.c_str(), res_name.c_str(), res_index, r.v[0],r.v[1],r.v[2]);
        }}

        pfile.close();
    }else if(mode=='a'){
        std::ofstream pfile(filename, std::ofstream::out|std::ofstream::app);
        pfile<<"MODEL\n";
        for(int i=0; i<groups.size(); i++){
            for(int j=0; j<groups[i]->atoms_global.size(); j++){
                cVector3 r;
                r = groups[i]->atoms_global[j];
                std::string atom_name;
                atom_name = groups[i]->atomNames[j];
                int atom_index = groups[i]->atomIndexes[j];
                int res_index = groups[i]->residueIndex;
                std::string res_name = convertRes1to3(groups[i]->residueName);
                pfile<<string_format("ATOM  %5d %4s %3s A%4d    %8.3f%8.3f%8.3f\n", atom_index, atom_name.c_str(), res_name.c_str(), res_index, r.v[0],r.v[1],r.v[2]);
        }}
        pfile<<"ENDMDL\n";
        pfile.close();
    }
}

