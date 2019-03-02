#ifndef CCONFORMATION_H_
#define CCONFORMATION_H_
#include <cVector3.h>
#include <cMatrix44.h>
#include <vector>
#include <string>
#include <cRigidGroup.h>
#include <cGeometry.h>

class cTransform{
    public:
        double *alpha, *beta;
        double *grad_alpha;
        double d;
        
    public:
        cMatrix44 mat, dmat;
        cTransform(double *param_alpha, double *param_beta, double d, double *grad_alpha){
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

class cNode{
    public:
        cRigidGroup *group;
        cTransform *T; //transform from parent to this node
        cMatrix44 M; //aggregate transform from root
        cMatrix44 F; //matrix for computing gradient
        cNode *left;
        cNode *right;
        cNode *parent;
                
        cNode(){parent = NULL;left=NULL;right=NULL;};
        ~cNode(){};
};

std::ostream& operator<<(std::ostream& os, const cNode& node);

class cConformation{
    // private:
    public:
        std::vector<cNode*> nodes;
        std::vector<cRigidGroup*> groups;
        std::vector<cTransform*> transforms;
        
        double zero_const;
        double *atoms_global; //pointer to computed coordinates
        cGeometry geo;
        uint num_atoms;

    public:
        cNode *root;

        // Construct protein graph and bind grad to the angles
        cConformation(std::string aa, double *angles, double *angles_grad, uint angles_length, double *atoms_global, bool add_terminal=false);    
        ~cConformation();
        
        void print(cNode *node);

        // Backward propagation with external gradients
        double backward(cNode *root_node, cNode *node, double *atoms_grad);
        void backward(cNode *node, double *atoms_grad);

        //Computes coordinates of the atoms
        void update(cNode *node);

        //Saving to pdb
        void save(std::string filename, const char mode);

    private:
        cNode *addNode(cNode *parent, cRigidGroup *group, cTransform *t);
        
        cNode *addGly(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addAla(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addSer(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addCys(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addVal(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addIle(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addLeu(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addThr(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addArg(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addLys(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addAsp(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addAsn(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addGlu(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addGln(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addMet(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addHis(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addPro(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addPhe(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addTyr(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
        cNode *addTrp(cNode *parentC, std::vector<double*> params, std::vector<double*> params_grad, bool terminal=false);
};

#endif