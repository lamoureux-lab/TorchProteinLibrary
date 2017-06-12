#include "cDensityMap.h"
#include <stdio.h>
#include <string>

cDensityMap::cDensityMap(THFloatTensor *t, cVector3 b0, float resolution){
    if(t->nDimension != 3 or (t->size[0]!=t->size[1] or t->size[0]!=t->size[3])){
        std::cout<<"cDensityMap::Invalid tensor"<<std::endl;
        return;
    }
    this->t = t;
    this->b0 = b0;
    this->resolution = resolution;
}
cDensityMap::~cDensityMap(){

}
    
void cDensityMap::saveAsXPlor(std::string filename){
    int size = t->size[0];
    float mean=0.5, std=0.5;
    
    FILE *fout = fopen(filename.c_str(), "w");
    fprintf(fout, "\n");
    fprintf(fout, " Density map\n");
    fprintf(fout, " 1\n");
    fprintf(fout, " 4\n");
    fprintf(fout, "%8d%8d%8d%8d%8d%8d%8d%8d%8d\n",size-1,0,size-1,size-1,0,size-1,size-1,0,size-1);
    fprintf(fout, "%12.5E%12.5E%12.5E%12.5E%12.5E%12.5E\n\n",size,size,size,90,90.,90.);
    fprintf(fout, "ZYX\n");
    for(int z=0; z<size; z++){
        fprintf(fout, "%8d\n", z-1);
        for(int y=0; y<size; y++){
            for(int x=0; x<size; x+=6){
                float val1 = THFloatTensor_get1d(t, x);
                float val2 = THFloatTensor_get1d(t, x+1);
                float val3 = THFloatTensor_get1d(t, x+2);
                float val4 = THFloatTensor_get1d(t, x+3);
                float val5 = THFloatTensor_get1d(t, x+4);
                float val6 = THFloatTensor_get1d(t, x+5);
                fprintf(fout, "%12.5E%12.5E%12.5E%12.5E%12.5E%12.5E\n", val1, val2, val3, val4, val5, val6);
            }
        }
    }
    fprintf(fout, "%8d\n", -9999);
    fprintf(fout, "%12.5E%12.5E\n", mean, std);
    fclose(fout);
    
}