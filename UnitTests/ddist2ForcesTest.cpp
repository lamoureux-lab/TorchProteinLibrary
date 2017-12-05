#include <TH.h>
#include <THC.h>
#include <GlutFramework.h>
#include <cVector3.h>
#include <Angles2CoordsAB/cAngles2CoordsAB.h>
#include <Angles2BMatrix/cAngles2BMatrix.h>
#include <Ddist2Forces/cDdist2Forces.h>

using namespace glutFramework;

void toGPUTensor(THCState*state, float *cpu_T, THCudaTensor *gpu_T){
    uint size = 1;
    for(int i=0; i<gpu_T->nDimension; i++)
        size *= gpu_T->size[i];
    cudaMemcpy( THCudaTensor_data(state, gpu_T), 
                cpu_T,
                size*sizeof(float), cudaMemcpyHostToDevice);
}

void toCPUTensor(THCState*state, THFloatTensor *cpu_T, THCudaTensor *gpu_T){
    uint size = 1;
    for(int i=0; i<gpu_T->nDimension; i++)
        size *= gpu_T->size[i];
    cudaMemcpy( THFloatTensor_data(cpu_T), 
                THCudaTensor_data(state, gpu_T),
                size*sizeof(float), cudaMemcpyDeviceToHost);
}

void toCPUTensor(THCState*state, THIntTensor *cpu_T, THCudaIntTensor *gpu_T){
    uint size = 1;
    for(int i=0; i<gpu_T->nDimension; i++)
        size *= gpu_T->size[i];
    cudaMemcpy( THIntTensor_data(cpu_T), 
                THCudaIntTensor_data(state, gpu_T),
                size*sizeof(int), cudaMemcpyDeviceToHost);
}

class ProteinVis: public Object{
    THCudaTensor *coords;
    THFloatTensor *cpu_coords;
    THCState* state;
    int num_atoms;
    
    public:
        ProteinVis(THCState* state, THCudaTensor *coords){
            this->coords = coords;
            this->state = state;
            num_atoms = coords->size[0]/3;
            cpu_coords = THFloatTensor_newWithSize1d(coords->size[0]);
        };
        ~ProteinVis(){
            THFloatTensor_free(cpu_coords);
        };
        void display(){
            toCPUTensor(state, cpu_coords, coords);
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

class ProteinMover: public Object{
    THCState* state;
    THCudaTensor *coords, *angles, *dangles, *A, *B, *ddist;
    THCudaTensor *force;
    cAngles2CoordsAB *a2c;
    cAngles2BMatrix *a2b;
    cDdist2Forces *dd2f;
    int num_angles, num_atoms;
    public:
        ProteinMover(THCState* state, int num_angles){
            this->state = state;
            this->num_angles = num_angles;
            this->num_atoms = num_angles+1;
            coords = THCudaTensor_newWithSize1d(state, num_atoms*3);
            force = THCudaTensor_newWithSize1d(state, num_atoms*3);
            angles = THCudaTensor_newWithSize2d(state, 2, num_angles);
            dangles = THCudaTensor_newWithSize2d(state, 2, num_angles);
            ddist = THCudaTensor_newWithSize2d(state, num_atoms, num_atoms);
            for(int i=0;i<num_angles;i++){
                THCudaTensor_set2d(state, angles, 0, i, 1.3);
                THCudaTensor_set2d(state, angles, 1, i, 0.5);
            }
            THCudaTensor_fill(state, ddist, 0.0);
            for(int i=0;i<num_atoms;i++){
                for(int j=i+1;j<num_atoms;j++){
                    float val = 0.01;
                    if(i==0 && j==9) val=0.01;
                    if(i==2 && j==5) val=0.01;
                    if(i==1 && j==4) val=-0.02;
                    THCudaTensor_set2d(state, ddist, i, j, val);
                    THCudaTensor_set2d(state, ddist, j, i, val);
                }
            }

            A = THCudaTensor_newWithSize1d(state, num_angles*16);
            this->a2c = new cAngles2CoordsAB(state, A, angles, NULL, num_angles);
            B = THCudaTensor_newWithSize2d(state, 2, num_angles*num_atoms*3);
            this->a2b = new cAngles2BMatrix(state, num_angles);
            this->dd2f = new cDdist2Forces(state, coords, num_angles);
        };
        ~ProteinMover(){
            THCudaTensor_free(state, coords);
            THCudaTensor_free(state, angles);
            THCudaTensor_free(state, dangles);
            THCudaTensor_free(state, ddist);
            THCudaTensor_free(state, A);
            THCudaTensor_free(state, B);
            delete a2c;
            delete a2b;
            delete dd2f;
        };
        THCudaTensor * get_coords(){
            return coords;
        };
        void display(){
            glPushAttrib(GL_LIGHTING_BIT);
            glDisable(GL_LIGHTING);
                        
            glLineWidth(3);
            glBegin(GL_LINES);
            
            THCudaTensor_fill(state, B, 0.0);
            THCudaTensor_fill(state, coords, 0.0);
            THCudaTensor_fill(state, dangles, 0.0);
            THCudaTensor_fill(state, force, 0.0);
            // THCudaTensor_fill(state, ddist, 0.0);

            a2c->computeForward(angles, coords);
            dd2f->computeForward(ddist, force);
            a2b->computeB(angles, coords, B);
            a2b->computeForward(force, dangles);
            
            for(int i=0; i<num_angles; i++){
                float alpha = THCudaTensor_get2d(state, angles, 0, i);
                float beta = THCudaTensor_get2d(state, angles, 1, i);
                alpha += THCudaTensor_get2d(state, dangles, 0, i);
                beta += THCudaTensor_get2d(state, dangles, 1, i);
                THCudaTensor_set2d(state, angles, 0, i, alpha);
                THCudaTensor_set2d(state, angles, 1, i, beta);
            }

            for(int i=0; i<num_atoms; i++){
                for(int j=i+1; j<num_atoms; j++){
                    float x0 = THCudaTensor_get1d(state, coords, 3*j);
                    float y0 = THCudaTensor_get1d(state, coords, 3*j+1);
                    float z0 = THCudaTensor_get1d(state, coords, 3*j+2);
                    float x1 = THCudaTensor_get1d(state, coords, 3*i);
                    float y1 = THCudaTensor_get1d(state, coords, 3*i+1);
                    float z1 = THCudaTensor_get1d(state, coords, 3*i+2);

                    if( fabs(THCudaTensor_get2d(state, ddist, i,j))>0.1){
                        if(THCudaTensor_get2d(state, ddist, i,j)>0)
                            glColor3f(1.0,0.0,0.0);
                        else
                            glColor3f(0.0,0.0,1.0);
                        glVertex3f(x0,y0,z0);
                        glVertex3f(x1,y1,z1);
                    }
                }          
            }       
            glColor3f(0.0,1.0,0.0);
            for(int j=0; j<num_atoms; j++){
                float fx = THCudaTensor_get1d(state, force, j*3);
                float fy = THCudaTensor_get1d(state, force, j*3+1);
                float fz = THCudaTensor_get1d(state, force, j*3+2);
                float x0 = THCudaTensor_get1d(state, coords, 3*j);
                float y0 = THCudaTensor_get1d(state, coords, 3*j+1);
                float z0 = THCudaTensor_get1d(state, coords, 3*j+2);
                float x1=x0+10*fx, y1=y0+10*fy, z1=z0+10*fz;
                if( (fx*fx + fy*fy +fz*fz)>0.01){
                    glVertex3f(x0,y0,z0);
                    glVertex3f(x1,y1,z1);         
                }          
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
    THCState* state = (THCState*) malloc(sizeof(THCState));
	memset(state, 0, sizeof(THCState));
	THCudaInit(state);
       
    ProteinMover pM(state, num_angles);
    ProteinVis pV(state, pM.get_coords());
    
    Vector<double> lookAtPos(0,0,0);
    framework.setLookAt(20.0, 20.0, 20.0, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);
    framework.addObject(&pM);
	framework.addObject(&pV);
    framework.startFramework(argc, argv);

    THCudaShutdown(state);
	free(state);
	
	return 0;
}
