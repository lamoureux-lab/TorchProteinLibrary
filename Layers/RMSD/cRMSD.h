#pragma once
#include <cVector3.h>
#include <cMatrix33.h>
#include <torch/torch.h>

template <typename T>
class cRMSD{
    private:
        bool external;

    public:
        cVector3<T> centroid_src, centroid_dst;
        uint num_atoms;

        T *ce_src, *ce_dst; //centered coordinates
        T *U_ce_src, *UT_ce_dst; //centered and rotated coordinates
        cMatrix33<T> U, UT; //rotation matrix and its transpose

    public:
        cRMSD(uint num_atoms);
        cRMSD(T *ce_src, T *ce_dst, T *U_ce_src, T *UT_ce_dst, const uint num_atoms);
        ~cRMSD();
        T compute( T *src, T *dst );
        void grad( T *grad_atoms, T *grad_output );
};