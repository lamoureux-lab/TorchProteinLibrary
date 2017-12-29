#include <TH.h>
#include <THC.h>
#include <GlutFramework.h>
#include <cVector3.h>

using namespace glutFramework;

class RigidGroupVis: public Object{
    cRigidGroup *rg;
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
                glColor3f(1.0,1.0,1.0);
                for(int i=0; i<rg->atoms.size(); i++){
                    float x = rg->atoms[i].v[0];
                    float y = rg->atoms[i].v[1];
                    float z = rg->atoms[i].v[2];
                    glVertex3f(x,y,z);
                }
            glEnd();
            glPopAttrib();
        };
};

class ConformationVis: public Object{
    cConformation *c;
    vector<RigidGroupVis> groupsVis;
    public:
        ConformationVis(cConformation *c){
            this->c = c;
            for(int i=0; i<c->groups.size(); i++){
                groupsVis.push_back(&(c->groups[i]));
            }
        };
        ~ConformationVis(){};
        void display(){
            for(int i=0; i<groupsVis.size(); i++){
                groupsVis[i].display();
            }
        };
};


int main(int argc, char** argv)
{
    int num_atoms = 10;
    int num_angles = num_atoms-1;
    GlutFramework framework;
    
    
    ConformationVis pV();
    
    Vector<double> lookAtPos(0,0,0);
    framework.setLookAt(20.0, 20.0, 20.0, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);
	framework.addObject(&pV);
    framework.startFramework(argc, argv);

    THCudaShutdown(state);
	free(state);
	
	return 0;
}
