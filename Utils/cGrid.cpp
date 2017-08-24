#include "cGrid.h"

template <class T>
cGrid<T>::cGrid(int num_bins, float resolution){
    this->num_bins = num_bins;
    this->resolution = resolution;
    b0 = cVector3(0,0,0);
    float end = num_bins*resolution;
    b1 = cVector3(end, end, end);

    array = new std::vector<T>[num_bins*num_bins*num_bins];
}

template <class T>
cGrid<T>::cGrid(int num_bins, cVector3 &b0, cVector3 &b1){
    this->num_bins = num_bins;
    this->b0 = b0;
    this->b1 = b1;
    this->resolution = (b1.v[0]-b1.v[1])/num_bins;

    array = new std::vector<T>[num_bins*num_bins*num_bins];
}

template <class T>
cGrid<T>::cGrid(float resolution, cVector3 &b0, cVector3 &b1){
    
    this->b0 = b0;
    this->b1 = b1;
    this->resolution = resolution;
    this->num_bins = int((b1.v[0]-b1.v[1])/resolution);

    array = new std::vector<T>[num_bins*num_bins*num_bins];
}

template <class T>
cGrid<T>::~cGrid(){
    delete [] array;
}

template <class T>
cVector3 cGrid<T>::idx2crd(cVector3 idx){
    cVector3 crd = idx*resolution + b0;
    return crd;
}

template <class T>
cVector3 cGrid<T>::crd2idx(cVector3 crd){
    cVector3 fidx = (crd - b0)/resolution;
    for(int i=0; i<3; i++)fidx.v[i]=int(fidx.v[i]);
    return fidx;
}

template <class T>
long cGrid<T>::idx2flat(cVector3 idx){
    long strides[] = {num_bins*num_bins, num_bins, 1};
    long flat_idx = 0;
    for(int i=0; i<3; i++) flat_idx += int(idx.v[i])*strides[i];
    return flat_idx;
}

template <class T>
void cGrid<T>::addObject(cVector3 crd, T* object){
    long cell_idx = idx2flat(crd2idx(idx));
    this->array[cell_idx].push_back(*object);
}

template <class T>
std::vector<T>* cGrid<T>::getObjects(cVector3 idx){
    long cell_idx = idx2flat(idx);
    return &(array[cell_idx]);
}

template <class T>
std::vector<T>* cGrid<T>::getObjects(int i){
    return &(array[i]);
}
