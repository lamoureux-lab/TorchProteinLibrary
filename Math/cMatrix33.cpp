
#include <math.h>

#include "cMatrix33.h"
#include "cVector3.h"


template <typename T> cMatrix33<T>::cMatrix33() {
	m[0][0]=m[0][1]=m[0][2]=m[1][0]=m[1][1]=m[1][2]=m[2][0]=m[2][1]=m[2][2]=0.0;
}

template <typename T> cMatrix33<T>::cMatrix33(T v) {
	m[0][0]=m[0][1]=m[0][2]=m[1][0]=m[1][1]=m[1][2]=m[2][0]=m[2][1]=m[2][2]=v;
}

template <typename T> cMatrix33<T>::cMatrix33(cVector3<T> d) {
	m[0][1]=m[0][2]=m[1][0]=m[1][2]=m[2][0]=m[2][1]=0.0;
	m[0][0]=d.v[0];m[1][1]=d.v[1];m[2][2]=d.v[2];
}

template <typename T> cMatrix33<T>::cMatrix33(T mat[3][3]) {
	m[0][0]=mat[0][0];m[0][1]=mat[0][1];m[0][2]=mat[0][2];
	m[1][0]=mat[1][0];m[1][1]=mat[1][1];m[1][2]=mat[1][2];
	m[2][0]=mat[2][0];m[2][1]=mat[2][1];m[2][2]=mat[2][2];
}

template <typename T> cMatrix33<T>::cMatrix33(T m00, T m01, T m02, T m10, T m11, T m12, T m20, T m21, T m22) {
	m[0][0]=m00;m[0][1]=m01;m[0][2]=m02;
	m[1][0]=m10;m[1][1]=m11;m[1][2]=m12;
	m[2][0]=m20;m[2][1]=m21;m[2][2]=m22;
}

template <typename T> cMatrix33<T>::cMatrix33(cVector3<T> v0, cVector3<T> v1, cVector3<T> v2) {
	m[0][0]=v0.v[0];m[1][0]=v0.v[1];m[2][0]=v0.v[2];
	m[0][1]=v1.v[0];m[1][1]=v1.v[1];m[2][1]=v1.v[2];
	m[0][2]=v2.v[0];m[1][2]=v2.v[1];m[2][2]=v2.v[2];
}

template <typename T> void cMatrix33<T>::setIdentity() {
	m[0][1]=m[0][2]=m[1][0]=m[1][2]=m[2][0]=m[2][1]=0.0;
	m[0][0]=m[1][1]=m[2][2]=1.0;
}

template <typename T> void cMatrix33<T>::setZero() {
	m[0][1]=m[0][2]=m[1][0]=m[1][2]=m[2][0]=m[2][1]=m[0][0]=m[1][1]=m[2][2]=0.0;
}

template <typename T> cMatrix33<T> cMatrix33<T>::operator+(cMatrix33& mat) {

	return cMatrix33(m[0][0]+mat.m[0][0],m[0][1]+mat.m[0][1],m[0][2]+mat.m[0][2],
		m[1][0]+mat.m[1][0],m[1][1]+mat.m[1][1],m[1][2]+mat.m[1][2],
		m[2][0]+mat.m[2][0],m[2][1]+mat.m[2][1],m[2][2]+mat.m[2][2]);

}

template <typename T> cMatrix33<T> cMatrix33<T>::operator-(cMatrix33& mat) {

	return cMatrix33(m[0][0]-mat.m[0][0],m[0][1]-mat.m[0][1],m[0][2]-mat.m[0][2],
		m[1][0]-mat.m[1][0],m[1][1]-mat.m[1][1],m[1][2]-mat.m[1][2],
		m[2][0]-mat.m[2][0],m[2][1]-mat.m[2][1],m[2][2]-mat.m[2][2]);

}

template <typename T> void cMatrix33<T>::operator+=(cMatrix33 &mat) {

	m[0][0]+=mat.m[0][0];m[0][1]+=mat.m[0][1];m[0][2]+=mat.m[0][2];
	m[1][0]+=mat.m[1][0];m[1][1]+=mat.m[1][1];m[1][2]+=mat.m[1][2];
	m[2][0]+=mat.m[2][0];m[2][1]+=mat.m[2][1];m[2][2]+=mat.m[2][2];

}

template <typename T> cMatrix33<T> cMatrix33<T>::operator*(const cMatrix33& mat) const {

	T res[3][3];
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

template <typename T> cMatrix33<T> cMatrix33<T>::getTranspose() const {

	T res[3][3];
	res[0][0]=m[0][0];res[0][1]=m[1][0];res[0][2]=m[2][0];
	res[1][0]=m[0][1];res[1][1]=m[1][1];res[1][2]=m[2][1];
	res[2][0]=m[0][2];res[2][1]=m[1][2];res[2][2]=m[2][2];

	return cMatrix33(res);

}

template <typename T> cVector3<T> cMatrix33<T>::operator*(const cVector3<T> &vec) const {

	T res[3];
	res[0]=m[0][0]*vec.v[0]+m[0][1]*vec.v[1]+m[0][2]*vec.v[2];
	res[1]=m[1][0]*vec.v[0]+m[1][1]*vec.v[1]+m[1][2]*vec.v[2];
	res[2]=m[2][0]*vec.v[0]+m[2][1]*vec.v[1]+m[2][2]*vec.v[2];

	return cVector3<T>(res[0], res[1], res[2]);

}

template <typename T> void cMatrix33<T>::operator*=(T d) {

	m[0][0]*=d;m[0][1]*=d;m[0][2]*=d;
	m[1][0]*=d;m[1][1]*=d;m[1][2]*=d;
	m[2][0]*=d;m[2][1]*=d;m[2][2]*=d;

}

template <typename T> cMatrix33<T> cMatrix33<T>::operator*(T d) const	{ return cMatrix33(m[0][0]*d,m[0][1]*d,m[0][2]*d,m[1][0]*d,m[1][1]*d,m[1][2]*d,m[2][0]*d,m[2][1]*d,m[2][2]*d); }


template <typename T> cMatrix33<T> cMatrix33<T>::diagMat(const cVector3<T>& diag)
{
    cMatrix33<T> M(0,0,0,0,0,0,0,0,0);
    for (std::size_t d=0;d<3;++d)
        {
        M.m[d][d] = diag[d];
        }

    return M;
}

template <typename T> cMatrix33<T> cMatrix33<T>::identity()
{
    cMatrix33<T> M(0,0,0,0,0,0,0,0,0);
    for (std::size_t d=0;d<3;++d)
        {
        M.m[d][d] = 1.0;
        }

    return M;
}

template <typename T> std::ostream& operator<<(std::ostream& out, const cMatrix33<T>& M)
{
    out << std::endl;
    out << M(0,0) << " " << M(0,1) << " " << M(0,2) << std::endl;
    out << M(1,0) << " " << M(1,1) << " " << M(1,2) << std::endl;
    out << M(2,0) << " " << M(2,1) << " " << M(2,2) << std::endl;

    return out;
}

template class cMatrix33<float>;
template std::ostream& operator<<(std::ostream&, const cMatrix33<float>&);

template class cMatrix33<double>;
template std::ostream& operator<<(std::ostream&, const cMatrix33<double>&);