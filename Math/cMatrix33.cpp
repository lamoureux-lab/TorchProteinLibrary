
#include <math.h>

#include "cMatrix33.h"
#include "cVector3.h"


cMatrix33::cMatrix33() {
	m[0][0]=m[0][1]=m[0][2]=m[1][0]=m[1][1]=m[1][2]=m[2][0]=m[2][1]=m[2][2]=0.0;
}

cMatrix33::cMatrix33(double v) {
	m[0][0]=m[0][1]=m[0][2]=m[1][0]=m[1][1]=m[1][2]=m[2][0]=m[2][1]=m[2][2]=v;
}

cMatrix33::cMatrix33(cVector3 d) {
	m[0][1]=m[0][2]=m[1][0]=m[1][2]=m[2][0]=m[2][1]=0.0;
	m[0][0]=d.v[0];m[1][1]=d.v[1];m[2][2]=d.v[2];
}

cMatrix33::cMatrix33(double mat[3][3]) {
	m[0][0]=mat[0][0];m[0][1]=mat[0][1];m[0][2]=mat[0][2];
	m[1][0]=mat[1][0];m[1][1]=mat[1][1];m[1][2]=mat[1][2];
	m[2][0]=mat[2][0];m[2][1]=mat[2][1];m[2][2]=mat[2][2];
}

cMatrix33::cMatrix33(double m00, double m01, double m02, double m10, double m11, double m12, double m20, double m21, double m22) {
	m[0][0]=m00;m[0][1]=m01;m[0][2]=m02;
	m[1][0]=m10;m[1][1]=m11;m[1][2]=m12;
	m[2][0]=m20;m[2][1]=m21;m[2][2]=m22;
}

cMatrix33::cMatrix33(cVector3 v0, cVector3 v1, cVector3 v2) {
	m[0][0]=v0.v[0];m[1][0]=v0.v[1];m[2][0]=v0.v[2];
	m[0][1]=v1.v[0];m[1][1]=v1.v[1];m[2][1]=v1.v[2];
	m[0][2]=v2.v[0];m[1][2]=v2.v[1];m[2][2]=v2.v[2];
}

void cMatrix33::setIdentity() {
	m[0][1]=m[0][2]=m[1][0]=m[1][2]=m[2][0]=m[2][1]=0.0;
	m[0][0]=m[1][1]=m[2][2]=1.0;
}

void cMatrix33::setZero() {
	m[0][1]=m[0][2]=m[1][0]=m[1][2]=m[2][0]=m[2][1]=m[0][0]=m[1][1]=m[2][2]=0.0;
}

cMatrix33 cMatrix33::operator+(cMatrix33& mat) {

	return cMatrix33(m[0][0]+mat.m[0][0],m[0][1]+mat.m[0][1],m[0][2]+mat.m[0][2],
		m[1][0]+mat.m[1][0],m[1][1]+mat.m[1][1],m[1][2]+mat.m[1][2],
		m[2][0]+mat.m[2][0],m[2][1]+mat.m[2][1],m[2][2]+mat.m[2][2]);

}

cMatrix33 cMatrix33::operator-(cMatrix33& mat) {

	return cMatrix33(m[0][0]-mat.m[0][0],m[0][1]-mat.m[0][1],m[0][2]-mat.m[0][2],
		m[1][0]-mat.m[1][0],m[1][1]-mat.m[1][1],m[1][2]-mat.m[1][2],
		m[2][0]-mat.m[2][0],m[2][1]-mat.m[2][1],m[2][2]-mat.m[2][2]);

}

void cMatrix33::operator+=(cMatrix33 &mat) {

	m[0][0]+=mat.m[0][0];m[0][1]+=mat.m[0][1];m[0][2]+=mat.m[0][2];
	m[1][0]+=mat.m[1][0];m[1][1]+=mat.m[1][1];m[1][2]+=mat.m[1][2];
	m[2][0]+=mat.m[2][0];m[2][1]+=mat.m[2][1];m[2][2]+=mat.m[2][2];

}

cMatrix33 cMatrix33::operator*(const cMatrix33& mat) const {

	double res[3][3];
	res[0][0]=m[0][0]*mat.m[0][0]+m[0][1]*mat.m[1][0]+m[0][2]*mat.m[2][0];
	res[0][1]=m[0][0]*mat.m[0][1]+m[0][1]*mat.m[1][1]+m[0][2]*mat.m[2][1];
	res[0][2]=m[0][0]*mat.m[0][2]+m[0][1]*mat.m[1][2]+m[0][2]*mat.m[2][2];

	res[1][0]=m[1][0]*mat.m[0][0]+m[1][1]*mat.m[1][0]+m[1][2]*mat.m[2][0];
	res[1][1]=m[1][0]*mat.m[0][1]+m[1][1]*mat.m[1][1]+m[1][2]*mat.m[2][1];
	res[1][2]=m[1][0]*mat.m[0][2]+m[1][1]*mat.m[1][2]+m[1][2]*mat.m[2][2];

	res[2][0]=m[2][0]*mat.m[0][0]+m[2][1]*mat.m[1][0]+m[2][2]*mat.m[2][0];
	res[2][1]=m[2][0]*mat.m[0][1]+m[2][1]*mat.m[1][1]+m[2][2]*mat.m[2][1];
	res[2][2]=m[2][0]*mat.m[0][2]+m[2][1]*mat.m[1][2]+m[2][2]*mat.m[2][2];

	return cMatrix33(res);

}

cMatrix33 cMatrix33::getTranspose() const {

	double res[3][3];
	res[0][0]=m[0][0];res[0][1]=m[1][0];res[0][2]=m[2][0];
	res[1][0]=m[0][1];res[1][1]=m[1][1];res[1][2]=m[2][1];
	res[2][0]=m[0][2];res[2][1]=m[1][2];res[2][2]=m[2][2];

	return cMatrix33(res);

}

cVector3 cMatrix33::operator*(const cVector3 &vec) const {

	double res[3];
	res[0]=m[0][0]*vec.v[0]+m[0][1]*vec.v[1]+m[0][2]*vec.v[2];
	res[1]=m[1][0]*vec.v[0]+m[1][1]*vec.v[1]+m[1][2]*vec.v[2];
	res[2]=m[2][0]*vec.v[0]+m[2][1]*vec.v[1]+m[2][2]*vec.v[2];

	return cVector3(res[0], res[1], res[2]);

}

void cMatrix33::operator*=(double d) {

	m[0][0]*=d;m[0][1]*=d;m[0][2]*=d;
	m[1][0]*=d;m[1][1]*=d;m[1][2]*=d;
	m[2][0]*=d;m[2][1]*=d;m[2][2]*=d;

}

cMatrix33 cMatrix33::operator*(double d) const	{ return cMatrix33(m[0][0]*d,m[0][1]*d,m[0][2]*d,m[1][0]*d,m[1][1]*d,m[1][2]*d,m[2][0]*d,m[2][1]*d,m[2][2]*d); }


cMatrix33 cMatrix33::diagMat(const cVector3& diag)
{
    cMatrix33 M(0,0,0,0,0,0,0,0,0);
    for (std::size_t d=0;d<3;++d)
        {
        M.m[d][d] = diag[d];
        }

    return M;
}

cMatrix33 cMatrix33::identity()
{
    cMatrix33 M(0,0,0,0,0,0,0,0,0);
    for (std::size_t d=0;d<3;++d)
        {
        M.m[d][d] = 1.0;
        }

    return M;
}

std::ostream& operator<<(std::ostream& out, const cMatrix33& M)
{
    out << std::endl;
    out << M(0,0) << " " << M(0,1) << " " << M(0,2) << std::endl;
    out << M(1,0) << " " << M(1,1) << " " << M(1,2) << std::endl;
    out << M(2,0) << " " << M(2,1) << " " << M(2,2) << std::endl;

    return out;
}
