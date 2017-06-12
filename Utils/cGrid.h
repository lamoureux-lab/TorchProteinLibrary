#ifndef CGRID_H_
#define CGRID_H_
#include <TH.h>
#include <THC.h>
#include <cVector3.h>
#include <vector>

template <class T>
class cGrid{
    
    std::vector<T>* array;

    int num_bins;
    float resolution;
    cVector3 b0, b1;

public:

	cGrid(int num_bins, float resolution);
    cGrid(int num_bins, cVector3 &b0, cVector3 &b1);
	~cGrid();
    
    cVector3 idx2crd(cVector3 idx);
    cVector3 crd2idx(cVector3 crd);
    long idx2flat(cVector3 idx);

    void addObject(cVector3 idx, T* object);
    std::vector<T>* getObject(cVector3 idx);

};

#endif