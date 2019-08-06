
#pragma once

#include "cVector3.h"

#include <math.h>

template <typename T>
class cMatrix33 {

public:

    T				m[3][3];

    cMatrix33();
    cMatrix33(T v);
    cMatrix33(cVector3<T> d);
    cMatrix33(T mat[3][3]);
    cMatrix33(T m00, T m01, T m02, T m10, T m11, T m12, T m20, T m21, T m22);
    cMatrix33(cVector3<T> v0, cVector3<T> v1, cVector3<T> v2);

    cMatrix33<T>			operator+(cMatrix33 &mat);
    cMatrix33<T>			operator-(cMatrix33 &mat);
    void				operator+=(cMatrix33 &mat);
    cMatrix33<T>			operator*(const cMatrix33 &mat) const;
    cVector3<T>			operator*(const cVector3<T> &vec) const;
    void				operator*=(T d);
    cMatrix33<T>			operator*(T d) const;

    void				setIdentity();
    void				setZero();
            
	cMatrix33<T>			getTranspose() const;

	void				computeEulerDecomposition(T &alpha, T &beta, T &gamma);

	inline const T& operator()(std::size_t i, std::size_t j) const { return m[i][j]; }
    inline       T& operator()(std::size_t i, std::size_t j)       { return m[i][j]; }

    static  cMatrix33<T>   diagMat(const cVector3<T>& diag);
    static  cMatrix33<T>   identity();
	

};

template <typename T> std::ostream& operator<<(std::ostream& out, const cMatrix33<T>& M);


