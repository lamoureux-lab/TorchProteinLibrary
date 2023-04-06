#include "cConformation.h"

#include <nUtil.h>
using namespace StringUtil;
using namespace ProtUtil;

template <typename T> void cTransform<T>::updateMatrix(){
    cMatrix44<T> Ry, Tr, Rx;
    Ry.setRy(*beta);
    Tr.setT(d, 'x');
    Rx.setRx(*alpha);
    mat = Ry*Tr*Rx; // Rx is dihedral angles (rotation/transform this)
}
template <typename T> void cTransform<T>::updateDMatrix(){
    cMatrix44<T> Ry, Tr, DRx;
    Ry.setRy(*beta);
    Tr.setT(d, 'x');
    DRx.setDRx(*alpha);
    dmat = Ry*Tr*DRx;
}
template <typename T> void cTransform<T>::print(){
    mat.print();
}


// template <typename T> std::ostream& operator<<(std::ostream& os, const cNode<T>& node){
//     return os<<*(node.group);
// }
// AS: need the function below
template <typename T> cConformation<T>::cConformation(std::string aa, T *angles, T *angles_grad, uint angles_length, T *atoms_global, int polymer_type, bool add_terminal){
    cNode<T> *lastC = NULL;
    zero_const = 0.0;
    this->atoms_global = atoms_global;
    bool terminal = false;
    if( polymer_type == 0){
    for(int i=0; i<aa.length(); i++){
        T *phi = angles + i + angles_length*0;T *dphi = angles_grad + i + angles_length*0;
        T *psi = angles + i + angles_length*1;T *dpsi = angles_grad + i + angles_length*1;
        
        T *omega, *domega;
        if(i>0){
            omega = angles + i-1 + angles_length*2;domega = angles_grad + i-1 + angles_length*2;
        }else{
            omega = &geo.omega_const;domega = NULL;
            // omega = &zero_const;domega = NULL;
        }

        T *xi1 = angles + i + angles_length*3;T *dxi1 = angles_grad + i + angles_length*3;
        T *xi2 = angles + i + angles_length*4;T *dxi2 = angles_grad + i + angles_length*4;
        T *xi3 = angles + i + angles_length*5;T *dxi3 = angles_grad + i + angles_length*5;
        T *xi4 = angles + i + angles_length*6;T *dxi4 = angles_grad + i + angles_length*6;
        T *xi5 = angles + i + angles_length*7;T *dxi5 = angles_grad + i + angles_length*7;
        std::vector<T*> params({phi, psi, omega, xi1, xi2, xi3, xi4, xi5});
        std::vector<T*> params_grad({dphi, dpsi, domega, dxi1, dxi2, dxi3, dxi4, dxi5});
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
    }
    if( polymer_type == 1){
    for(int i=0; i<aa.length(); i++){
        T *alpha = angles + i + angles_length*0;T *dalpha = angles_grad + i + angles_length*0;
        T *beta = angles + i + angles_length*1;T *dbeta = angles_grad + i + angles_length*1;
        T *gamma = angles + i + angles_length*2;T *dgamma = angles_grad + i + angles_length*2;
        T *delta = angles + i + angles_length*3;T *ddelta = angles_grad + i + angles_length*3;
        T *epsilon = angles + i + angles_length*4;T *depsilon = angles_grad + i + angles_length*4;
        T *zeta = angles + i + angles_length*5;T *dzeta = angles_grad + i + angles_length*5;
        T *nu0 = angles + i + angles_length*6;T *dnu0 = angles_grad + i + angles_length*6;
        T *nu1 = angles + i + angles_length*7;T *dnu1 = angles_grad + i + angles_length*7;
        T *nu2 = angles + i + angles_length*8;T *dnu2 = angles_grad + i + angles_length*8;
        T *nu3 = angles + i + angles_length*9;T *dnu3 = angles_grad + i + angles_length*9;
        T *nu4 = angles + i + angles_length*10;T *dnu4 = angles_grad + i + angles_length*10;
        T *chi = angles + i + angles_length*11;T *dchi = angles_grad + i + angles_length*11;
//        T *gamma, *dgamma;
//        if(i>0){
//            gamma = angles + i-1 + angles_length*2;dgamma = angles_grad + i-1 + angles_length*2;
//        }else{
//            gamma = &geo.omega_const;dgamma = NULL;
//            // omega = &zero_const;domega = NULL;
//        }

        T *xi1 = angles + i + angles_length*12;T *dxi1 = angles_grad + i + angles_length*12;
        T *xi2 = angles + i + angles_length*13;T *dxi2 = angles_grad + i + angles_length*13;
        T *xi3 = angles + i + angles_length*14;T *dxi3 = angles_grad + i + angles_length*14;
        T *xi4 = angles + i + angles_length*15;T *dxi4 = angles_grad + i + angles_length*15;
        T *xi5 = angles + i + angles_length*16;T *dxi5 = angles_grad + i + angles_length*16;
        T *xi6 = angles + i + angles_length*17;T *dxi6 = angles_grad + i + angles_length*17;
        T *xi7 = angles + i + angles_length*18;T *dxi7 = angles_grad + i + angles_length*18;
        T *xi8 = angles + i + angles_length*19;T *dxi8 = angles_grad + i + angles_length*19;
        T *xi9 = angles + i + angles_length*20;T *dxi9 = angles_grad + i + angles_length*20;
        T *xi10 = angles + i + angles_length*21;T *dxi10 = angles_grad + i + angles_length*21;
        T *xi11 = angles + i + angles_length*22;T *dxi11 = angles_grad + i + angles_length*22;
        T *xi12 = angles + i + angles_length*23;T *dxi12 = angles_grad + i + angles_length*23;
        std::vector<T*> params({alpha, beta, gamma, delta, epsilon, zeta, nu0, nu1, nu2, nu3, nu4, chi, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9, xi10, xi11, xi12});
        std::vector<T*> params_grad({dalpha, dbeta, dgamma, ddelta, depsilon, dzeta, dnu0, dnu1, dnu2, dnu3, dnu4, dchi, dxi1, dxi2, dxi3, dxi4, dxi5, dxi6, dxi7, dxi8, dxi9, dxi10, dxi11, dxi12});
        if(add_terminal){
            if(i == (aa.length()-1))
                terminal = true;
            else
                terminal = false;
        }
        switch(aa[i]){
            std::cout << "switch(aa[i])" << (aa[i]);
            case 'G':
                lastC = addDG(lastC, params, params_grad, terminal); //addDG
                std::cout << "addDG called" << "\n";
                break;
            case 'A':
                lastC = addDA(lastC, params, params_grad, terminal); //addDA
                std::cout << "addDA called" << "\n";
                break;
            case 'T':
                lastC = addDT(lastC, params, params_grad, terminal); //addDT
                std::cout << "addDT called" << "\n";
                break;
            case 'C':
                lastC = addDC(lastC, params, params_grad, terminal); //addDC
                std::cout << "addDC called" << "\n";
                break;
        }
    }
    std::cout << "Conformation loop finished";
    }
    
    //Computing conformation
    this->update(this->root);
    //Computing number of atoms
    this->num_atoms = 0;
    for(int i=0; i<groups.size();i++){
        num_atoms += groups[i]->atoms_global.size();
    }
}

template <typename T> cConformation<T>::~cConformation(){
    for(int i=0; i<nodes.size(); i++)
        delete nodes[i];
    for(int i=0; i<transforms.size(); i++)
        delete transforms[i];
    for(int i=0; i<groups.size(); i++)
        delete groups[i];
}
//AS: The two function for the graphs.(addNode, and update node, backward)
template <typename T> cNode<T> *cConformation<T>::addNode(cNode<T> *parent, cRigidGroup<T> *group, cTransform<T> *t){
    cNode<T> *new_node = new cNode<T>();
    new_node->group = group;
    new_node->Tr = t;
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

template <typename T> void cConformation<T>::update(cNode<T> *node){
    node->Tr->updateMatrix();
    node->Tr->updateDMatrix();
    if(node->parent!=NULL){    
        node->M = (node->parent->M) * (node->Tr->mat);
        node->F = (node->parent->M) * (node->Tr->dmat) * invertTransform44(node->M);
        node->group->applyTransform(node->M);
    }else{
        node->M.setIdentity();
        node->F = (node->Tr->dmat) * invertTransform44(node->M);
        node->group->applyTransform(node->M);
    }
    if(node->left!=NULL){
        update(node->left);
    }
    if(node->right!=NULL){
        update(node->right);
    }
}

template <typename T> T cConformation<T>::backward(cNode<T> *root_node, cNode<T> *node, T *atoms_grad){
    T grad = 0.0;
    if(node->left!=NULL){
        grad += backward(root_node, node->left, atoms_grad);
    }
    if(node->right!=NULL){
        grad += backward(root_node, node->right, atoms_grad);
    }
    for(int i=0;i<node->group->atoms_global.size(); i++){
        cVector3<T> gradVec(
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


template <typename T> void cConformation<T>::backward(cNode<T> *node, T *atoms_grad){
    for(int i=0; i<nodes.size();i++){
        if(nodes[i]->Tr->grad_alpha!=NULL) 
            *(nodes[i]->Tr->grad_alpha) = backward(nodes[i], nodes[i], atoms_grad);
    }
}

template <typename T> void cConformation<T>::print(cNode<T> *node){
    // std::cout<<*node;
    // if(node->left!=NULL){
    //     std::cout<<"---";
    //     print(node->left);
    // }
    // if(node->right!=NULL){
    //     std::cout<<"|\n";
    //     std::cout<<"---";
    //     print(node->right);
    // }
    // std::cout<<".\n";
    
}
//AS: function to save output
template <typename T> void cConformation<T>::save(std::string filename, const char mode){
    if(mode=='w'){
        std::ofstream pfile(filename, std::ofstream::out);
    
        for(int i=0; i<groups.size(); i++){
            for(int j=0; j<groups[i]->atoms_global.size(); j++){
                cVector3<T> r;
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
                cVector3<T> r;
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

template class cConformation<float>;
template class cConformation<double>;