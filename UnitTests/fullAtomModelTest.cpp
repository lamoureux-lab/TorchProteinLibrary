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
            glPointSize(5);
            glBegin(GL_POINTS);
                
                for(int i=0; i<rg->atoms.size(); i++){
                    float x = rg->atoms[i].v[0];
                    float y = rg->atoms[i].v[1];
                    float z = rg->atoms[i].v[2];
                    if(rg->atomNames[i]=="C" || rg->atomNames[i]=="CB"){
                        glColor3f(0.0,0.5,0.0);
                        glVertex3f(x,y,z);
                    }
                    if(rg->atomNames[i]=="O"){
                        glColor3f(0.8,0.,0.);
                        glVertex3f(x,y,z);
                    }
                    if(rg->atomNames[i]=="CA"){
                        glColor3f(0.0,0.8,0.0);
                        glVertex3f(x,y,z);
                    }
                    if(rg->atomNames[i]=="N"){
                        glColor3f(0,0,0.8);
                        glVertex3f(x,y,z);
                    }
                    
                }
            glEnd();
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
        void display(){
            for(int i=0; i<groupsVis.size(); i++){
                groupsVis[i].display();
            }
            glPushAttrib(GL_LIGHTING_BIT);
            glDisable(GL_LIGHTING);
            glLineWidth(2);
            glBegin(GL_LINES);
                glColor3f(1.0,1.0,1.0);
                for(int i=0; i<groupsVis.size()-1; i++){
                    // cVector3 x0 = groupsVis[i].rg->atoms[0];
                    // cVector3 x1 = groupsVis[i+1].rg->atoms[0];
                    // glVertex3f(x0.v[0],x0.v[1],x0.v[2]);
                    // glVertex3f(x1.v[0],x1.v[1],x1.v[2]);
                }
            glEnd();
            glPopAttrib();
        };
};


int main(int argc, char** argv)
{
    int num_atoms = 10;
    int num_angles = num_atoms-1;
    GlutFramework framework;
    
    double alpha[3];
    double beta[3];
    std::string aa("AAA");
    for(int i=0;i<3;i++){
        alpha[i] = -1.047;
        beta[i] = -0.698;
    }
    
    cConformation conf(aa, &alpha[0], &beta[0]);
    std::cout<<"lalala"<<std::endl;
    conf.print( conf.root );
    conf.update( conf.root );
    ConformationVis pV(&conf);
    
    Vector<double> lookAtPos(0,0,0);
    framework.setLookAt(20.0, 20.0, 20.0, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);
	framework.addObject(&pV);
    framework.startFramework(argc, argv);

	return 0;
}
