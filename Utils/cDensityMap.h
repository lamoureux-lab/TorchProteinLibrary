#ifndef CDENSITYMAP_H_
#define CDENSITYMAP_H_
#include <TH.h>
#include <cVector3.h>
#include <vector>

class cDensityMap{
    
    THFloatTensor *t;
    float resolution;
    cVector3 b0;

public:

	cDensityMap(THFloatTensor *t, cVector3 b0, float resolution);
	~cDensityMap();
    
    void saveAsXPlor(std::string filename);
};

#endif