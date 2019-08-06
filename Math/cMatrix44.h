#pragma once

#include "cVector3.h"
#include "cMatrix33.h"

#include <math.h>

template <typename T>
class cMatrix44 {
private:
    bool external;
    T	*m = NULL;
public:

    cMatrix44();
    cMatrix44(const cMatrix44<T>& other);
    ~cMatrix44();
    cMatrix44(T *mat);
    cMatrix44(T mat[4][4]);
    cMatrix44(const cMatrix33<T> &rot, const cVector3<T> &shift);
    
    void setDihedral(const T phi, const T psi, const T R);
    void setDihedralDphi(const T phi, const T psi, const T R);
    
    void setRx(const T angle);
    void setRy(const T angle);
    void setRz(const T angle);
    void setDRx(const T angle);
    void setT(const T R, const char axis);
    void setIdentity();
    
    cMatrix44<T>		operator*(const cMatrix44<T> &mat) const;
    cVector3<T>			operator*(const cVector3<T> &vec) const;
    void				operator*=(T d);
    
        
    void				print();   
	
    inline const    T& operator()(std::size_t i, std::size_t j) const { return m[4*i+j]; }
    inline          T& operator()(std::size_t i, std::size_t j)       { return m[4*i+j]; }
    void            operator=(const cMatrix44<T>& u);
};


template <typename T> cMatrix44<T> invertTransform44(const cMatrix44<T> &mat);