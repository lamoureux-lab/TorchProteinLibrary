#include <TH.h>
#include <GlutFramework.h>
#include <cVector3.h>
#include <cMarchingCubes.h>
#include <iostream>

using namespace glutFramework;

extern "C" {

    int VisualizeVolume4d(THFloatTensor *cpu_volume){
        GlutFramework framework;
        if(cpu_volume->nDimension!=4){
            std::cout<<"VisualizeVolume4d: nDimension should be 4"<<std::endl;
            return -1;
        }
        int spatial_dim = cpu_volume->size[1];
        
        cVolume v(cpu_volume);
        Vector<double> lookAtPos(spatial_dim/2,spatial_dim/2,spatial_dim/2);
        framework.setLookAt(spatial_dim, spatial_dim, spatial_dim, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);
        framework.addObject(&v);
        
        framework.startFramework(0, NULL);     
    }
}