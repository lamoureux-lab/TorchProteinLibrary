#include <GlutFramework.h>
#include <cVector3.h>
#include <cRigidGroup.h>
#include <cConformation.h>
#include <string>
#include <iostream>

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
        void display(){
            glPushAttrib(GL_LIGHTING_BIT);
            glDisable(GL_LIGHTING);
            glPointSize(8.0);
            glBegin(GL_POINTS);
                
                for(int i=0; i<rg->atoms_global.size(); i++){
                    float x = rg->atoms_global[i].v[0];
                    float y = rg->atoms_global[i].v[1];
                    float z = rg->atoms_global[i].v[2];
                    if(rg->atomNames[i][0]=='C' && rg->atomNames[i]!="CA"){
                        glColor3f(0.0,0.5,0.0);
                        glVertex3f(x,y,z);
                    }
                    if(rg->atomNames[i][0]=='O'){
                        glColor3f(0.8,0.,0.);
                        glVertex3f(x,y,z);
                    }
                    if(rg->atomNames[i]=="CA"){
                        glColor3f(0.0,0.8,0.0);
                        glVertex3f(x,y,z);
                    }
                    if(rg->atomNames[i][0]=='N'){
                        glColor3f(0,0,0.8);
                        glVertex3f(x,y,z);
                    }
                    if(rg->atomNames[i][0]=='S'){
                        glColor3f(0.9,0.9,0.0);
                        glVertex3f(x,y,z);
                    }
                    
                }
            glEnd();
            if(rg->atoms_global.size()>1){
                glBegin(GL_LINE_LOOP);
                    glColor3f(0.5,0.5,0.5);
                    for(int i=0; i<rg->atoms_global.size(); i++){
                        float x = rg->atoms_global[i].v[0];
                        float y = rg->atoms_global[i].v[1];
                        float z = rg->atoms_global[i].v[2];
                        glVertex3f(x,y,z);
                    }
                glEnd();
            }
            
            glPopAttrib();
        };
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
        void walk(cNode *node){
            if(node->left!=NULL){
                cVector3 x0 = node->group->atoms_global[0];
                cVector3 x1 = node->left->group->atoms_global[0];
                glVertex3f(x0.v[0],x0.v[1],x0.v[2]);
                glVertex3f(x1.v[0],x1.v[1],x1.v[2]);
                walk(node->left);
            }
            if(node->right!=NULL){
                cVector3 x0 = node->group->atoms_global[0];
                cVector3 x1 = node->right->group->atoms_global[0];
                glVertex3f(x0.v[0],x0.v[1],x0.v[2]);
                glVertex3f(x1.v[0],x1.v[1],x1.v[2]);
                walk(node->right);
            }
        }
        void display(){
            glPushAttrib(GL_LIGHTING_BIT);
            glDisable(GL_LIGHTING);
            glLineWidth(2);
            glBegin(GL_LINES);
                glColor3f(1.0,1.0,1.0);
                walk(c->root);
            glEnd();
            glPopAttrib();

            for(int i=0; i<groupsVis.size(); i++){
                groupsVis[i].display();
            }
        };
};


class ConformationUpdate: public Object{
    cConformation *c;
    double *data;
    int length;
    public:
        ConformationUpdate(cConformation *c, double *data, int length){
            this->c = c;
            this->data = data;
            this->length = length;
        };
        ~ConformationUpdate(){};
        void display(){
            for(int i=0;i<length;i++){
                data[2*length + i]+=0.01;
            }
            c->update(c->root);
        };
};


int main(int argc, char** argv)
{
    GlutFramework framework;
    
    std::string aa("GASCV");

    int length = aa.length();
    int num_angles = 4;
    double th_data[length*num_angles];
        
    for(int i=0;i<length;i++){
        th_data[i + length*0] = -1.047;
        th_data[i + length*1] = -0.698;
        th_data[i + length*2] = 110.4*M_PI/180.0;
        th_data[i + length*3] = -63.3*M_PI/180.0;
    }
    
    cConformation conf(aa, &th_data[0], length);
    std::cout<<"lalala"<<std::endl;
    conf.print( conf.root );
    conf.update( conf.root );
    ConformationVis pV(&conf);
    ConformationUpdate cU(&conf, th_data, length);
    
    Vector<double> lookAtPos(0,0,0);
    framework.setLookAt(20.0, 20.0, 20.0, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);
	framework.addObject(&pV);
    framework.addObject(&cU);
    framework.startFramework(argc, argv);

	return 0;
}
