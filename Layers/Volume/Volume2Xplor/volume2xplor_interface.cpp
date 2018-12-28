#include "volume2xplor_interface.h"
#include <stdio.h>
#include <string>

void Volume2Xplor( at::Tensor volume, const char *filename, float resolution){
    // if( (volume.type().is_cuda()) || volume.dtype() != at::kFloat ){
    //     throw("Incorrect tensor types");
    //     std::cout<<"Incorrect tensor types"<<std::endl;
    // }
    if(volume.ndimension() != 3){
        std::cout<<"Incorrect input ndim"<<std::endl;
        throw("Incorrect input ndim");
    }
    auto V = volume.accessor<float, 3>();
    int size = volume.size(0);
    float mean=0.5, std=0.5;
    
    FILE *fout = fopen(filename, "w");
    fprintf(fout, "\n");
    fprintf(fout, " Density map\n");
    fprintf(fout, " 1\n");
    fprintf(fout, " 0\n");
    fprintf(fout, "%8d%8d%8d%8d%8d%8d%8d%8d%8d\n",size-1,0,size-1,size-1,0,size-1,size-1,0,size-1);
    fprintf(fout, "%12.5E%12.5E%12.5E%12.5E%12.5E%12.5E\n",float(size)*resolution,float(size)*resolution,float(size)*resolution,90.,90.,90.);
    fprintf(fout, "ZYX\n");
    
    for(int z=0; z<size; z++){
        fprintf(fout, "%8d\n", z);
        unsigned long int ind=1;
        for(int y=0; y<size; y++){
            for(int x=0; x<size; x++, ind++){
                fprintf(fout, "%12.5E", V[x][y][z]);
                if(ind%6 == 0){
                    fprintf(fout, "\n");
                }
            }
        }
        if(ind%6 != 0){
            for(int i=0; i<(6 - ind%6 + 1); i++)fprintf(fout, "%12.5E", 0.0);
            fprintf(fout, "\n");
        }
    }
    fprintf(fout, "%8d\n", -9999);
    fprintf(fout, "%12.5E%12.5E\n", mean, std);
    fclose(fout);

}

/*
void Volume2Xplor(  at::Tensor volume, const char *filename){
    if( (volume.type().is_cuda()) || volume.dtype() != at::kFloat ){
        throw("Incorrect tensor types");
        std::cout<<"Incorrect tensor types"<<std::endl;
    }
    if(volume.ndimension() != 3){
        std::cout<<"Incorrect input ndim"<<std::endl;
        throw("Incorrect input ndim");
    }
    auto V = volume.accessor<float, 3>();
    int size = volume.size(0);
    float mean=0.5, std=0.5;
    
    FILE *fout = fopen(filename, "wb");
    // fprintf(fout, "\n");
    fprintf(fout, "%80s", "Density map");
    fprintf(fout, "%80s", "3");
    fprintf(fout, "%80s", "4");
    fprintf(fout, "%80s", "5");
    fprintf(fout, "%8d%8d%8d%8d%8d%8d%8d%8d%8d",size-1,0,size-1,size-1,0,size-1,size-1,0,size-1);
    fprintf(fout, "%12.5E%12.5E%12.5E%12.5E%12.5E%12.5E",float(size),float(size),float(size),90.,90.,90.);
    fprintf(fout, "ZYX");
    unsigned long int ind=1;
    for(int z=0; z<size; z++){
        fprintf(fout, "%8d", z);
        for(int y=0; y<size; y++){
            for(int x=0; x<size; x++, ind++){
                fprintf(fout, "%12.5E", V[x][y][z]);
            }
        }
    }
    fprintf(fout, "%8d", -9999);
    fprintf(fout, "%12.5E%12.5E", mean, std);
    fclose(fout);

}
*/