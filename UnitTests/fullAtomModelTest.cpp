#include <TH.h>
#include <THC.h>
#include <GlutFramework.h>
#include <cVector3.h>

using namespace glutFramework;

class ProteinVis: public Object{
    
    public:
        ProteinVis(){
        };
        ~ProteinVis(){
        
        };
        void display(){
            glPushAttrib(GL_LIGHTING_BIT);
            glDisable(GL_LIGHTING);
            glPointSize(5);
            glBegin(GL_POINTS);
                glColor3f(1.0,0.0,0.0);
                for(int i=0; i<num_atoms; i++){
                    float x = THFloatTensor_get1d(cpu_coords, 3*i);
                    float y = THFloatTensor_get1d(cpu_coords, 3*i+1);
                    float z = THFloatTensor_get1d(cpu_coords, 3*i+2);
                    glVertex3f(x,y,z);
                }
            glEnd();

            glLineWidth(3);
            glBegin(GL_LINES);
                glColor3f(1.0,1.0,1.0);
                for(int i=0; i<(num_atoms-1); i++){
                    float x0 = THFloatTensor_get1d(cpu_coords, 3*i);
                    float y0 = THFloatTensor_get1d(cpu_coords, 3*i+1);
                    float z0 = THFloatTensor_get1d(cpu_coords, 3*i+2);
                    float x1 = THFloatTensor_get1d(cpu_coords, 3*(i+1));
                    float y1 = THFloatTensor_get1d(cpu_coords, 3*(i+1)+1);
                    float z1 = THFloatTensor_get1d(cpu_coords, 3*(i+1)+2);
                    glVertex3f(x0,y0,z0);
                    glVertex3f(x1,y1,z1);
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
       
    ProteinVis pV();
    
    Vector<double> lookAtPos(0,0,0);
    framework.setLookAt(20.0, 20.0, 20.0, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);
	framework.addObject(&pV);
    framework.startFramework(argc, argv);

    THCudaShutdown(state);
	free(state);
	
	return 0;
}
