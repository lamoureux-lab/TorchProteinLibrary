#pragma once

#include "cVector3.h"
#include "cMatrix33.h"

#include <math.h>

class cMatrix44 {
private:
    bool external;
    double	*m = NULL;
public:

    cMatrix44();
    cMatrix44(const cMatrix44& other);
    ~cMatrix44();
    cMatrix44(double *mat);
    cMatrix44(double mat[4][4]);
    cMatrix44(const cMatrix33 &rot, const cVector3 &shift);
    
    void setDihedral(const double phi, const double psi, const double R);
    void setDihedralDphi(const double phi, const double psi, const double R);
    
    void setRx(const double angle);
    void setRy(const double angle);
    void setRz(const double angle);
    void setDRx(const double angle);
    void setT(const double R, const char axis);
    void setIdentity();
    
    cMatrix44			operator*(const cMatrix44 &mat) const;
    cVector3			operator*(const cVector3 &vec) const;
    void				operator*=(double d);
    
        
    void				print();   
	
    inline const    double& operator()(std::size_t i, std::size_t j) const { return m[4*i+j]; }
    inline          double& operator()(std::size_t i, std::size_t j)       { return m[4*i+j]; }
    void            operator=(const cMatrix44& u);
};


cMatrix44 invertTransform44(const cMatrix44 &mat);