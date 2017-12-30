#ifndef CCONFORMATION_H_
#define CCONFORMATION_H_
#include <cVector3.h>
#include <cMatrix44.h>
#include <vector>
#include <string>
#include <cRigidGroup.h>

class cTransform{
    private:
        double *alpha, *beta;
        double d;
        
        cMatrix44 getRx(double angle);
        cMatrix44 getRy(double angle);
        cMatrix44 getRz(double angle);
        cMatrix44 getT(double dist, char axis);
    public:
        cMatrix44 mat;
        cTransform(double *param_alpha, double *param_beta, double d){
            this->alpha = param_alpha;
            this->beta = param_beta;
            this->d = d;
        };
        ~cTransform(){};
        void updateMatrix();
        void print();
};

class cNode{
    public:
        cRigidGroup *group;
        cTransform *T; //transform from parent to this node
        cMatrix44 M; //aggregate transform from root
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
        
        double zero_const, omega_const, kappa1, kappa2, kappa3;

    public:
        cNode *root;

        cConformation(std::string aa, double *alpha, double *beta);
        ~cConformation();
        cNode *addNode(cNode *parent, cRigidGroup *group, cTransform *t);
        void update(cNode *node);
        void print(cNode *node);
};

#endif