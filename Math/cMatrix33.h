
#pragma once

#include "cVector3.h"

#include <math.h>

class cMatrix33 {

public:

    double				m[3][3];

    cMatrix33();
    cMatrix33(double v);
    cMatrix33(cVector3 d);
    cMatrix33(double mat[3][3]);
    cMatrix33(double m00, double m01, double m02, double m10, double m11, double m12, double m20, double m21, double m22);
    cMatrix33(cVector3 v0, cVector3 v1, cVector3 v2);

    cMatrix33			operator+(cMatrix33 &mat);
    cMatrix33			operator-(cMatrix33 &mat);
    void				operator+=(cMatrix33 &mat);
    cMatrix33			operator*(const cMatrix33 &mat) const;
    cVector3			operator*(const cVector3 &vec) const;
    void				operator*=(double d);
    cMatrix33			operator*(double d) const;

    void				setIdentity();
    void				setZero();
            
	cMatrix33			getTranspose() const;

	void				computeEulerDecomposition(double &alpha, double &beta, double &gamma);

	inline const double& operator()(std::size_t i, std::size_t j) const { return m[i][j]; }
    inline       double& operator()(std::size_t i, std::size_t j)       { return m[i][j]; }

    static  cMatrix33   diagMat(const cVector3& diag);
    static  cMatrix33   identity();
	

};

std::ostream& operator<<(std::ostream& out, const cMatrix33& M);

typedef cMatrix33 *pcMatrix33;


