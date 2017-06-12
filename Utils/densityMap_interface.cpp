#include <TH.h>
#include <THC.h>
#include <iostream>
#include <string>
#include <cDensityMap.h>
#include <cVector3.h>

extern THCState *state;

extern "C" {
    int SaveVolumeXPlor(THFloatTensor *cpu_volume, const char* filename){
        cDensityMap dmap(cpu_volume, cVector3(0.0,0.0,0.0), 1.0);
        dmap.saveAsXPlor(std::string(filename));
    }
}