#ifndef CCONFORMATION_H_
#define CCONFORMATION_H_
#include <cVector3.h>
#include <cMatrix44.h>
#include <vector>
#include <string>
#include <cRigidGroup.h>
#include <cGeometry.h>
#include <torch/extension.h>
//AS: function for graphs
template <typename T>
class cTransform{
    public:
        T *alpha, *beta;
        T *grad_alpha;
        T d;
        
    public:
        cMatrix44<T> mat, dmat;
        cTransform(T *param_alpha, T *param_beta, T d, T *grad_alpha){
            this->alpha = param_alpha;
            this->beta = param_beta;
            this->d = d;
            this->grad_alpha = grad_alpha;
        };
        ~cTransform(){};
        void updateMatrix();
        void updateDMatrix();
        void print();
};

template <typename T>
class cNode{
    public:
        cRigidGroup<T> *group;
        cTransform<T> *Tr; //transform from parent to this node
        cMatrix44<T> M; //aggregate transform from root
        cMatrix44<T> F; //matrix for computing gradient
        cNode *left;
        cNode *right;
        cNode *parent;
                
        cNode(){parent = NULL;left=NULL;right=NULL;};
        ~cNode(){};
};

// template <typename T> std::ostream& operator<<(std::ostream& os, const cNode<T>& node);

template <typename T>
class cConformation{
    // private:
    public:
        std::vector<cNode<T>*> nodes;
        std::vector<cRigidGroup<T>*> groups;
        std::vector<cTransform<T>*> transforms;
        
        T zero_const;
        T *atoms_global; //pointer to computed coordinates
        cGeometry<T> geo;
        uint num_atoms;

    public:
        cNode<T> *root;

        // Construct protein graph and bind grad to the angles
        cConformation(std::string aa, T *angles, T *angles_grad, uint angles_length, T *atoms_global, int polymer_type = 0, torch::Tensor chain_names = {}, bool add_terminal=false);
        ~cConformation();
        
        void print(cNode<T> *node);

        // Backward propagation with external gradients
        T backward(cNode<T> *root_node, cNode<T> *node, T *atoms_grad);
        void backward(cNode<T> *node, T *atoms_grad);

        //Computes coordinates of the atoms
        void update(cNode<T> *node);

        //Saving to pdb
        void save(std::string filename, const char mode);

    private:
        cNode<T> *addNode(cNode<T> *parent, cRigidGroup<T> *group, cTransform<T> *t);
        
        cNode<T> *addGly(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addAla(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addSer(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addCys(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addVal(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addIle(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addLeu(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addThr(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addArg(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addLys(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addAsp(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addAsn(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addGlu(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addGln(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addMet(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addHis(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addPro(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addPhe(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addTyr(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addTrp(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, bool terminal=false);
        cNode<T> *addDG(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addDA(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addDT(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addDC(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addDG_5Prime(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addDA_5Prime(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addDT_5Prime(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addDC_5Prime(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addG(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addA(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addU(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addC(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addG_5Prime(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addA_5Prime(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addU_5Prime(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
        cNode<T> *addC_5Prime(cNode<T> *parentC, std::vector<T*> params, std::vector<T*> params_grad, char last_res);
};

#endif