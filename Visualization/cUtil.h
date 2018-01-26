#pragma once

#include <PracticalSocket.h>  // For Socket, ServerSocket, and SocketException
#include <iostream>           // For cout, cerr
#include <cstdlib>            // For atoi()  
#include <pthread.h>          // For POSIX threads  
#include <GlutFramework.h>
#include <cVector3.h>
#include <cRigidGroup.h>
#include <cConformation.h>
#include <cPDBLoader.h>
#include <string>
#include <iostream>
#include <thread>

using namespace glutFramework;

class RigidGroupVis: public Object{
    cRigidGroup *rg;
    friend class ConformationVis;
    public:
        
        RigidGroupVis(cRigidGroup *rg){
            this->rg = rg;
        };
        ~RigidGroupVis(){
        
        };
        void display();
        
};

class ConformationVis: public Object{
    cConformation *c;
    std::vector<RigidGroupVis> groupsVis;
    public:
        ConformationVis(cConformation *c){
            
            this->c = c;
            for(int i=0; i<c->groups.size(); i++){
                RigidGroupVis vis(c->groups[i]);
                groupsVis.push_back(vis);
            }
        };
        ~ConformationVis(){};
        void walk(cNode *node);
        void display();
};

class ConformationUpdate: public Object{
    cConformation *c;
    public:
        ConformationUpdate(cConformation *c){
            this->c = c;
        };
        ~ConformationUpdate(){};
        void display(){
            c->update(c->root);
        };
};