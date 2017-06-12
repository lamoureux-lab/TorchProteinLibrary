#include <TH/TH.h>
#include <cDensityMap.h>
#include <cVector3.h>
#include <math.h>

int main(int argc, char** argv)
{
    int spatial_dim = 120;
    THFloatTensor *grid = THFloatTensor_newWithSize3d(spatial_dim,spatial_dim,spatial_dim);
    for(int x=0; x<spatial_dim; x++){
        for(int y=0; y<spatial_dim; y++){
            for(int z=0; z<spatial_dim; z++){
                float c = spatial_dim/2.0;
                float r = sqrt((x-c)*(x-c) + (y-c)*(y-c) + (z-c)*(z-c));
                THFloatTensor_set3d(grid, x,y,z, sin(r/(0.1*c)));
            }
        }
    }
    cDensityMap dm(grid, cVector3(0,0,0), 1.0);
    dm.saveAsXPlor("test.xplor");
}