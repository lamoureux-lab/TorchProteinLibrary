#ifndef CSO3SAMPLER_H_
#define CSO3SAMPLER_H_
#include <cVector3.h>
#include <cMatrix33.h>
#include <vector>
#include <iostream>

//This code adapted from https://github.com/mlund/situs.git

class cSO3Sampler{
    public:
        cSO3Sampler(double dAngle);
        ~cSO3Sampler(){return;};
        std::vector<cMatrix33> U;
    private:
        void remap_eulers (double *psi_out, double *theta_out, double *phi_out,
                        double psi_in, double theta_in, double phi_in,
                        double psi_ref, double theta_ref, double phi_ref);

        void eu_spiral (double eu_range[3][2], double delta, unsigned long *eu_count, float  *&eu_store);
        cMatrix33 convertZXZtoU (double psi, double theta, double phi);
};

#endif