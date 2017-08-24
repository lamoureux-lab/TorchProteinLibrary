#pragma once
#include <TH.h>
#include <THC.h>

class cCoords2RMSD{

    THCState *state;
    THCudaTensor *re_coordinates_src, *re_coordinates_dst; //centered coordinates
    THCudaTensor *U_coordinates_src, *Ut_coordinates_dst; //transformed and centered coordinates
    // THCudaTensor *T; //T-matrix
    THCudaTensor *rot_mat, *rot_mat_t; //T-matrix
    THCudaTensor *centroid_src, *centroid_dst; //centroids
    // THFloatTensor *vec, *eig; //eigenvalues and vectors
    
    float U[9], Ut[9]; // rotation matrix
    int angles_length;
    float rmsd;

    public:
    cCoords2RMSD(   THCState *state,
                    THCudaTensor *re_coordinates_src, 
                    THCudaTensor *re_coordinates_dst,
                    THCudaTensor *U_coordinates_src,
                    THCudaTensor *Ut_coordinates_dst,
                    THCudaTensor *rot_mat_t, int length);

    ~cCoords2RMSD();
    float computeForward(THCudaTensor *coordinates_src, THCudaTensor *coordinates_dst);
    void computeBackward(THCudaTensor *gradInput, float gradOutput);

};