#pragma once
#include <cVector3.h>
#include <cMatrix33.h>

class cRMSD{
    private:
        bool external;

    public:
        cVector3 centroid_src, centroid_dst;
        uint num_atoms;

        double *ce_src, *ce_dst; //centered coordinates
        double *U_ce_src, *UT_ce_dst; //centered and rotated coordinates
        cMatrix33 U, UT; //rotation matrix and its transpose

    public:
        cRMSD(uint num_atoms);
        cRMSD(double *ce_src, double *ce_dst, double *U_ce_src, double *UT_ce_dst, uint num_atoms);
        ~cRMSD();
        double compute( double *src, double *dst );
        void grad( double *grad_atoms, double *grad_output );
};