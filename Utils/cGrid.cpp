#include "cGrid.h"

template <class T>
cGrid::cGrid(int num_bins, float resolution){
    this->num_bins = num_bins;
    this->resolution = resolution;
    b0 = cVector3(0,0,0);
    float end = num_bins*resolution;
    b1 = cVector3(end, end, end);

    array = new std::vector<T>[num_bins*num_bins*num_bins];
}

template <class T>
cGrid::cGrid(int num_bins, cVector3 &b0, cVector3 &b1){
    this->num_bins = num_bins;
    this->b0 = b0;
    this->b1 = b1;
    this->resolution = (b1.v[0]-b1.v[1])/num_bins;

    array = new std::vector<T>[num_bins*num_bins*num_bins];
}

template <class T>
cGrid::~cGrid(){
    delete [] array;
}

template <class T>
cVector3 cGrid::idx2crd(cVector3 idx){
    cVector3 crd = idx*resolution + b0;
    return crd;
}

template <class T>
cVector3 cGrid::crd2idx(cVector3 crd){
    cVector3 fidx = (crd - b0)/resolution;
    for(int i=0; i<3; i++)fidx.v[i]=int(fidx.v[i]);
    return fidx;
}

template <class T>
long cGrid::idx2flat(cVector3 idx){
    long strides = {num_bins*num_bins, num_bins, 1};
    long flat_idx = 0;
    for(int i=0; i<3; i++) flat_idx += int(idx.v[i])*strides[i];
    return flat_idx;
}

template <class T>
void cGrid::addObject(cVector3 idx, T* object){
    long cell_idx = idx2flat(idx);
    this->array[cell_idx].push_back(*object);
}

template <class T>
std::vector<T>* cGrid::getObject(cVector3 idx){
    long cell_idx = idx2flat(idx);
    return &(array[cell_idx]);
}
