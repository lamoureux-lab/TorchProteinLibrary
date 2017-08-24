#include <TH.h>
#include <THC.h>
#include <GlutFramework.h>
#include <cAngles2CoordsDihedral.h>

using namespace glutFramework;

int main(int argc, char** argv)
{

    GlutFramework framework;
    THCState* state = (THCState*) malloc(sizeof(THCState));
	memset(state, 0, sizeof(THCState));
	THCudaInit(state); 
       
    THFloatTensor *angles = THFloatTensor_newWithSize2d(2, 8);
    THFloatTensor_set2d(angles, 0, 0, 0);
    THFloatTensor_set2d(angles, 1, 0, -0.7453119990371283);  

    THFloatTensor_set2d(angles, 0, 1, -1.0929153369806561);
    THFloatTensor_set2d(angles, 1, 1, -0.71983838445421366);
    
    THFloatTensor_set2d(angles, 0, 2, -1.0904891667387324);
    THFloatTensor_set2d(angles, 1, 2, -0.7485);

    THFloatTensor_set2d(angles, 0, 3, -1.1484);
    THFloatTensor_set2d(angles, 1, 3, -0.693685);
    
    THFloatTensor_set2d(angles, 0, 4, -1.081996);
    THFloatTensor_set2d(angles, 1, 4, -0.72569);

    THFloatTensor_set2d(angles, 0, 5, -1.081996);
    THFloatTensor_set2d(angles, 1, 5, -0.72569);

    THFloatTensor_set2d(angles, 0, 6, -1.081996);
    THFloatTensor_set2d(angles, 1, 6, -0.72569);


    THFloatTensor_set2d(angles, 0, 7, -1.04316);
    THFloatTensor_set2d(angles, 1, 7, 0.0);


    cAngles2CoordsDihedral c2b(state, angles);

    Vector<double> lookAtPos(0,0,0);
    framework.setLookAt(10, 10.0, 10.0, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);
	framework.addObject(&c2b);
    framework.startFramework(argc, argv);

    THCudaShutdown(state);
	free(state);
	
	return 0;
}