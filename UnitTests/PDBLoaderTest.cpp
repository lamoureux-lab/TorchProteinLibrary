#include <GlutFramework.h>
#include <cVector3.h>
#include <cPDBLoader.h>
#include <string>
#include <iostream>

using namespace glutFramework;

class PDBVis: public Object{
    cPDBLoader *pdb;
    public:
        PDBVis(cPDBLoader *pdb){
            this->pdb = pdb;
        };
        ~PDBVis(){};
        void display(){
            
            glPushAttrib(GL_LIGHTING_BIT);
            glDisable(GL_LIGHTING);
            glLineWidth(2);
            glBegin(GL_POINTS);
                glColor3f(1.0,.0,.0);
                for(int i=0;i<pdb->res_res_names.size();i++){
                    for(int j=0;j<pdb->res_r[i].size();j++){
                        cVector3 x0, x1;
                        x0 = pdb->res_r[i][j];
                        glVertex3f(x0.v[0],x0.v[1],x0.v[2]);
                    }
                }
            glEnd();
            glBegin(GL_LINES);
                glColor3f(1.0,1.0,1.0);
                for(int i=0;i<pdb->res_res_names.size();i++){
                    for(int j=0;j<pdb->res_r[i].size();j++){
                        for(int k=0;k<pdb->res_r[i].size();k++){
                            cVector3 x0, x1;
                            x0 = pdb->res_r[i][j];
                            x1 = pdb->res_r[i][k];
                            if( (x0 - x1).norm2()<4.0 ){
                                glVertex3f(x0.v[0],x0.v[1],x0.v[2]);
                                glVertex3f(x1.v[0],x1.v[1],x1.v[2]);
                            }
                        }
                    }
                }
            glEnd();
            glPopAttrib();
        };
};


int main(int argc, char** argv)
{
    GlutFramework framework;
    
    double th_atoms[1309*3];
    
    cPDBLoader pdb(std::string("2lzm.pdb"));
    std::cout<<"loaded"<<std::endl;
    try{
        pdb.reorder(th_atoms);
    }catch(std::string msg){
        std::cout<<msg<<std::endl;
    }
    PDBVis pdbVis(&pdb);
    Vector<double> lookAtPos(0,0,0);
    framework.setLookAt(20.0, 20.0, 20.0, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);
    framework.addObject(&pdbVis);
	framework.startFramework(argc, argv);

	return 0;
}
