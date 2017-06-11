/*************************************************************************\

  Copyright 2004 The University of North Carolina at Chapel Hill.
  All Rights Reserved.

  Permission to use, copy, modify OR distribute this software and its
  documentation for educational, research and non-profit purposes, without
  fee, and without a written agreement is hereby granted, provided that the
  above copyright notice and the following three paragraphs appear in all
  copies.

  IN NO EVENT SHALL THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL BE
  LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
  CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
  USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY
  OF NORTH CAROLINA HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH
  DAMAGES.

  THE UNIVERSITY OF NORTH CAROLINA SPECIFICALLY DISCLAIM ANY
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
  PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
  NORTH CAROLINA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

  The author may be contacted via:

  US Mail:             Stephane Redon
                       Department of Computer Science
                       Sitterson Hall, CB #3175
                       University of N. Carolina
                       Chapel Hill, NC 27599-3175

  Phone:               (919) 962-1930

  EMail:               redon@cs.unc.edu

\**************************************************************************/


#include <math.h>

#include "cMatrix33.h"
#include "cVector3.h"

#ifndef M_PI_2
#define M_PI_2 (3.14159265358979323846/2.0)
#endif

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

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

	return cVector3(res);

}

void cMatrix33::operator*=(double d) {

	m[0][0]*=d;m[0][1]*=d;m[0][2]*=d;
	m[1][0]*=d;m[1][1]*=d;m[1][2]*=d;
	m[2][0]*=d;m[2][1]*=d;m[2][2]*=d;

}

cMatrix33 cMatrix33::operator*(double d) const	{ return cMatrix33(m[0][0]*d,m[0][1]*d,m[0][2]*d,m[1][0]*d,m[1][1]*d,m[1][2]*d,m[2][0]*d,m[2][1]*d,m[2][2]*d); }

void cMatrix33::toQuaternion(double &w, double &x, double &y, double &z) {

	// conversion to a quaternion

	double w2=0.25 * (1. + m[0][0] + m[1][1] + m[2][2]);
	const double zero=1e-40;

	if (w2 > zero)
	{
		w=sqrt(w2);
		x=(m[2][1] - m[1][2]) / (4.0*w);
		y=(m[0][2] - m[2][0]) / (4.0*w);
		z=(m[1][0] - m[0][1]) / (4.0*w);

/* 		w2=0.25 * (1. + m[0][0] - m[1][1] - m[2][2]);
 * 		w=sqrt(w2);
 * 		x=(m[1][0] + m[0][1]) / (4.0*w);
 * 		y=(m[0][2] + m[2][0]) / (4.0*w);
 * 		z=(m[1][2] - m[2][1]) / (4.0*w);
 */
	}
	else
	{
		w=0.;
		double x2=0.5 * (m[1][1] + m[2][2]);
		if (x2 > zero)
		{
			x=sqrt(x2);
			y=m[1][0] / (2.0*x);
			z=m[2][0] / (2.0*x);
		}
		else
		{
			x=0.;
			double y2=0.5 * (1. - m[2][2]);
			if (y2 > zero)
			{
				y=sqrt(y2);
				z=m[2][1] / (2.0*y);
			}
			else
			{
				y=0.;
				z=1.;
			}
		}
	}

	return;
	// normalize quaternion

	double invnormq=1.0/sqrt(w*w+x*x+y*y+z*z);
	w*=invnormq;
	x*=invnormq;
	y*=invnormq;
	z*=invnormq;

}

void cMatrix33::orthonormalize() {

	// This function normalizes the links orientations, i.e. makes sure that the orientation matrices are orthonormal (to floating point precision)

	// conversion to a quaternion

	double x, y, z, w;
	double w2=0.25 * (1. + m[0][0] + m[1][1] + m[2][2]);
	const double zero=0.00000001;

	if (w2 > zero)
	{
		w=sqrt(w2);
		x=(m[2][1] - m[1][2]) / (4.0*w);
		y=(m[0][2] - m[2][0]) / (4.0*w);
		z=(m[1][0] - m[0][1]) / (4.0*w);
	}
	else
	{
		w=0.;
		double x2=0.5 * (m[1][1] + m[2][2]);
		if (x2 > zero)
		{
			x=sqrt(x2);
			y=m[1][0] / (2.0*x);
			z=m[2][0] / (2.0*x);
		}
		else
		{
			x=0.;
			double y2=0.5 * (1. - m[2][2]);
			if (y2 > zero)
			{
				y=sqrt(y2);
				z=m[2][1] / (2.0*y);
			}
			else
			{
				y=0.;
				z=1.;
			}
		}
	}

	// normalize quaternion

	double invnormq=1.0/sqrt(w*w+x*x+y*y+z*z);
	w*=invnormq;
	x*=invnormq;
	y*=invnormq;
	z*=invnormq;

	// convert back to orientation matrix

	m[0][0]=1.0 - 2.0 * y * y - 2.0 * z * z;
	m[0][1]=2.0 * x * y - 2.0 * w * z;
	m[0][2]=2.0 * x * z + 2.0 * w * y;
	m[1][0]=2.0 * x * y + 2.0 * w * z;
	m[1][1]=1.0 - 2.0 * x * x - 2.0 * z * z;
	m[1][2]=2.0 * y * z - 2.0 * w * x;
	m[2][0]=2.0 * x * z - 2.0 * w * y;
	m[2][1]=2.0 * y * z + 2.0 * w * x;
	m[2][2]=1.0 - 2.0 * x * x - 2.0 * y * y;

}

void cMatrix33::computeEulerDecomposition(double &alpha, double &beta, double &gamma) {

	// perform an Euler decomposition of a matrix assumed to be a rotation matrix
	//
	// R=R(z,alpha)R(y,beta)R(z,gamma)

	// NB: the returned value of beta is always positive

	double sinBeta=sqrt(1-m[2][2]*m[2][2]);

	if (sinBeta>0.00000001) {

		alpha=atan2(-m[1][2],-m[0][2]);
		beta=atan2(sinBeta,m[2][2]);
		gamma=atan2(-m[2][1],m[2][0]);

	}
	else {

		alpha=atan2(m[1][0],m[0][0]);
		beta=0.0;
		gamma=0.0;

	}

}

void cMatrix33::computeEulerDecompositionZYZ(double &phi, double &theta, double &psi) {

	// perform an Euler decomposition of a matrix assumed to be a rotation matrix
	//
	// R=R(z,psi)R(y,theta)R(z,phi)

	// NB: the returned value of beta is always positive
	// rotation axis are fixed
	double cosTheta2 = m[2][2]*m[2][2];
	double sinTheta= cosTheta2 > 1? 0:sqrt(1-cosTheta2);

	//std::cout << "cos,sin theta= " << m[2][2] <<" "<< sinTheta <<"\n";
	double phiPlusPsi, phiMinPsi;


	if (sinTheta>0.00000001) {

		//psi=atan2(m[1][2],-m[0][2]);
		theta=atan2(sinTheta,m[2][2]);
		phi=atan2(m[2][1],m[2][0]);

	}
	else {
		//std::cout << "else\n";
		theta=0.0;
		phi=0.0;
		//psi=atan2(-m[1][0],m[0][0]);


		if (m[2][2] < 0) {
			theta=M_PI;
			phi=0.0;
			//psi=atan2(m[1][0],-m[0][0]);
		}


	}

	// this modifocation gives right results at sin(theta) approx 0
	// otherwise there can be problems with phi+psi shifted by Pi
	if (m[2][2] > 0.0) {
		phiPlusPsi=atan2( m[0][1]-m[1][0], m[0][0]+m[1][1]);
		psi=phiPlusPsi - phi;
	}
	else {
		phiMinPsi=atan2(-( m[0][1]+m[1][0]), -(m[0][0]-m[1][1]));
		psi=phi - phiMinPsi;
	}

}

void cMatrix33::print()	{

	double SMALL_VALUE=10e-200;
	for (int i=0;i<3;i++) {

		for (int j=0;j<3;j++) {
			if (fabs(m[i][j]) < SMALL_VALUE && fabs(m[i][j]) > 0.0)
				std::cout << "\t[" << "O" << "]";
			else {
				std::cout << "\t[" << m[i][j] << "]";
			}
		}
		std::cout << std::endl;

	}

	std::cout << std::endl;
	std::cout << std::endl;

}

cMatrix33 cMatrix33::fromAxisAngle(const cVector3& rkAxis, double fRadians) {

	cMatrix33 res;

	double fCos=cos(fRadians);
	double fSin=sin(fRadians);
	double fOneMinusCos=1.0 - fCos;
	double fX2=rkAxis.v[0] * rkAxis.v[0];
	double fY2=rkAxis.v[1] * rkAxis.v[1];
	double fZ2=rkAxis.v[2] * rkAxis.v[2];
	double fXYM=rkAxis.v[0] * rkAxis.v[1] * fOneMinusCos;
	double fXZM=rkAxis.v[0] * rkAxis.v[2] * fOneMinusCos;
	double fYZM=rkAxis.v[1] * rkAxis.v[2] * fOneMinusCos;
	double fXSin=rkAxis.v[0] * fSin;
	double fYSin=rkAxis.v[1] * fSin;
	double fZSin=rkAxis.v[2] * fSin;

	res.m[0][0]=fX2 * fOneMinusCos + fCos;
	res.m[0][1]=fXYM - fZSin;
	res.m[0][2]=fXZM + fYSin;
	res.m[1][0]=fXYM + fZSin;
	res.m[1][1]=fY2 * fOneMinusCos + fCos;
	res.m[1][2]=fYZM - fXSin;
	res.m[2][0]=fXZM - fYSin;
	res.m[2][1]=fYZM + fXSin;
	res.m[2][2]=fZ2 * fOneMinusCos + fCos;

	return res;

}

cMatrix33 cMatrix33::fromAxisAnglePi(const cVector3& rkAxis) {

	cMatrix33 res;

	double fX2=rkAxis.v[0] * rkAxis.v[0];
	double fY2=rkAxis.v[1] * rkAxis.v[1];
	double fZ2=rkAxis.v[2] * rkAxis.v[2];
	double fXYM=rkAxis.v[0] * rkAxis.v[1] * 2;
	double fXZM=rkAxis.v[0] * rkAxis.v[2] * 2;
	double fYZM=rkAxis.v[1] * rkAxis.v[2] * 2;

	res.m[0][0]=fX2 * 2.0 -1.0;
	res.m[0][1]=fXYM;
	res.m[0][2]=fXZM ;
	res.m[1][0]=fXYM ;
	res.m[1][1]=fY2 * 2.0 -1.0;
	res.m[1][2]=fYZM;
	res.m[2][0]=fXZM ;
	res.m[2][1]=fYZM ;
	res.m[2][2]=fZ2 * 2.0 -1.0;

	return res;

}

void cMatrix33::revoluteJointZAxisMatrix(cVector3 &axis) {

	// this function returns an orthonormal basis with a given z axis

	cVector3 basisZ(axis);
	basisZ.normalize();
	cVector3 basisY;
	if ((fabs(axis.v[1])>0.1)||(fabs(axis.v[2])>0.1)) basisY=basisZ^cVector3(1,0,0);
	else basisY=basisZ^cVector3(0,1,0);

	basisY.normalize();
	cVector3 basisX(basisY^basisZ);

	*this=cMatrix33(basisX,basisY,basisZ);

}

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

#include <cmath>

cMatrix33 cMatrix33::getSquareRoot() const
{
    cVector3  lambda;
    cMatrix33 Q;

    diagonalize(lambda, Q);

    cMatrix33 QT = Q.getTranspose();
    cVector3  sqrtLambda(std::sqrt(lambda[0]), std::sqrt(lambda[1]), std::sqrt(lambda[2]));

    return Q*diagMat(sqrtLambda)*QT;
}

cMatrix33 cMatrix33::getInverseSquareRoot() const
{
    cVector3  lambda;
    cMatrix33 Q;

    diagonalize(lambda, Q);

    cMatrix33 QT = Q.getTranspose();
    cVector3  iSqrtLambda(1.0 / std::sqrt(lambda[0]), 1.0 / std::sqrt(lambda[1]), 1.0 / std::sqrt(lambda[2]));
    
    return Q*diagMat(iSqrtLambda)*QT;
}

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);a[k][l]=h+s*(g-h*tau);
#define cdfabs(x) ((x<0)? -x : x)

void cMatrix33::diagonalize(cVector3 &ev, cMatrix33 &p) const {

	// NB : M=Pdiag(e)PINVERSE (et PINVERSE=PTRANSPOSE)
	// NB : From Numerical Recipes

	int nrot; // number of Jacobi rotations
	int i,j,ip,iq;
	double tresh, theta,tau,t,sm,s,h,g,c;

	double a[3][3];
	double v[3][3];
	double d[3];
	double b[3];
	double z[3];

	// initialization

	for (i=0;i<3;i++) {
		for (j=0;j<3;j++) {
			a[i][j]=m[i][j];
			v[i][j]=0.0;
		}
		v[i][i]=1.0;
	}

	for (i=0;i<3;i++) {
		b[i]=a[i][i];
		d[i]=a[i][i];
		z[i]=0.0;
	}

	nrot=0;

	for (i=1;i<=50;i++) {

		sm=fabs(a[0][1])+fabs(a[0][2])+fabs(a[1][2]);

		if (sm==0.0) {

			for (ip=0;ip<3;ip++) {
				for (iq=0;iq<3;iq++) p.m[ip][iq]=v[ip][iq];
				ev.v[ip]=d[ip];
			}

			return;

		}

		if (i<4) tresh=0.2*sm/9.0;
		else tresh=0.0;

		for (ip=0;ip<2;ip++) {
			for (iq=ip+1;iq<3;iq++) {
				g=100.0*cdfabs(a[ip][iq]);
				if ((i>4)&&(cdfabs(d[ip]+g)==cdfabs(d[ip]))&&(cdfabs(d[iq]+g)==cdfabs(d[iq])))
					a[ip][iq]=0;
				else if (cdfabs(a[ip][iq])>tresh) {
					h=d[iq]-d[ip];
					if ((cdfabs(h)+g)==(cdfabs(h))) t=a[ip][iq]/h;
					else {
						theta=0.5*h/(a[ip][iq]);
						t=1.0/(cdfabs(theta)+sqrt(1.0+theta*theta));
						if (theta<0.0) t=-t;
					}
					c=1.0/sqrt(1+t*t);
					s=t*c;
					tau=s/(1.0+c);
					h=t*a[ip][iq];
					z[ip]-=h;
					z[iq]+=h;
					d[ip]-=h;
					d[iq]+=h;
					a[ip][iq]=0.0;
					for (j=0;j<=(ip-1);j++) {
						ROTATE(a,j,ip,j,iq)
					}
					for (j=(ip+1);j<=(iq-1);j++) {
						ROTATE(a,ip,j,j,iq)
					}
					for (j=(iq+1);j<3;j++) {
						ROTATE(a,ip,j,iq,j)
					}
					for (j=0;j<3;j++) {
						ROTATE(v,j,ip,j,iq)
					}
					nrot++;
				}
			}
		}
		for (ip=0;ip<3;ip++) {
			b[ip]+=z[ip];
			d[ip]=b[ip];
			z[ip]=0.0;
		}
	}

	std::cout << "Error in diagonalization in Matrix33" << std::endl;

}


double cMatrix33::det() const {

	return (m[0][0]*m[1][1]*m[2][2]+m[1][0]*m[2][1]*m[0][2]+m[2][0]*m[0][1]*m[1][2])-(m[2][0]*m[1][1]*m[0][2]+m[0][0]*m[2][1]*m[1][2]+m[1][0]*m[0][1]*m[2][2]);

}

/// norm2 of the matrix the hard way
double cMatrix33::norm2() const {

	//compute maximum eigenvalue
	cMatrix33 AtA=getTranspose()*(*this);
	cVector3 eV;
	cMatrix33 P;
	AtA.diagonalize(eV,P);
	double rho=0.0;
	if(rho<fabs(eV.v[0])) rho=fabs(eV.v[0]);
	if(rho<fabs(eV.v[1])) rho=fabs(eV.v[1]);
	if(rho<fabs(eV.v[2])) rho=fabs(eV.v[2]);

	return sqrt(rho);

}

/// if is a rotation matrix, its angle
double cMatrix33::cosphi() const {

	//compute maximum eigenvalue
	cMatrix33 A=*this;
	cVector3 eV;
	cMatrix33 P;
	A.diagonalize(eV,P);
	double rho=1.0;
	if(rho>fabs(eV.v[0])) rho=(eV.v[0]);
	if(rho>fabs(eV.v[1])) rho=(eV.v[1]);
	if(rho>fabs(eV.v[2])) rho=(eV.v[2]);

	return rho;

}

cMatrix33 cMatrix33::Inverse() const {

	cMatrix33 mat=*this;

	//compute determinant
	double det;
	det=(m[0][0]*m[1][1]*m[2][2]+m[1][0]*m[2][1]*m[0][2]+m[2][0]*m[0][1]*m[1][2])-(m[2][0]*m[1][1]*m[0][2]+m[0][0]*m[2][1]*m[1][2]+m[1][0]*m[0][1]*m[2][2]);
	//if(fabs(det)<1e-5)
	//	int dummy=0;


	mat=mat.getTranspose();

	double res[3][3];
	res[0][0]=(mat.m[1][1]*mat.m[2][2]-mat.m[2][1]*mat.m[1][2])/det;
	res[0][1]=-(mat.m[1][0]*mat.m[2][2]-mat.m[2][0]*mat.m[1][2])/det;
	res[0][2]=(mat.m[1][0]*mat.m[2][1]-mat.m[2][0]*mat.m[1][1])/det;
	res[1][0]=-(mat.m[0][1]*mat.m[2][2]-mat.m[2][1]*mat.m[0][2])/det;
	res[1][1]=(mat.m[0][0]*mat.m[2][2]-mat.m[2][0]*mat.m[0][2])/det;
	res[1][2]=-(mat.m[0][0]*mat.m[2][1]-mat.m[2][0]*mat.m[0][1])/det;
	res[2][0]=(mat.m[0][1]*mat.m[1][2]-mat.m[1][1]*mat.m[0][2])/det;
	res[2][1]=-(mat.m[0][0]*mat.m[1][2]-mat.m[1][0]*mat.m[0][2])/det;
	res[2][2]=(mat.m[0][0]*mat.m[1][1]-mat.m[1][0]*mat.m[0][1])/det;

	return cMatrix33(res);

}

cVector3 cMatrix33::rotateToAxis( const cVector3 &axis) {
	/* rotates the matrix to a given axis over Z and Y axes
	  * by phi and theta angles
	  */

	cVector3 	zPrime, result;
	double 		cos_theta, mod_sin_theta, cos_phi, sin_phi;
	double SMALL_VALUE=1e-10;

	zPrime=axis.normalizedVersion();

	cos_theta=zPrime.v[2];
	mod_sin_theta=sqrt(zPrime.v[0]*zPrime.v[0] + zPrime.v[1]*zPrime.v[1]);

	/* ZYZ convention */
	if (fabs(zPrime.v[0]) < SMALL_VALUE && fabs(zPrime.v[1]) < SMALL_VALUE) {
		//phi is undefined
		cos_phi=1.0;
		sin_phi=0.0;
	}
	else if (fabs(zPrime.v[0]) < SMALL_VALUE ) {
		cos_phi=0.0;
		if (zPrime.v[1] < 0.0)
			sin_phi=-1.0;
		else
			sin_phi=1.0;
	}
	else if (fabs(zPrime.v[1]) < SMALL_VALUE ) {
		sin_phi=0.0;
		if (zPrime.v[0] < 0.0)
			cos_phi=-1.0;
		else
			cos_phi=1.0;
	}
	else {
		double temp=zPrime.v[0]/zPrime.v[1];
		double temp2=temp*temp;
		cos_phi=1.0/sqrt(1.0+1.0/temp2);
		sin_phi=1.0/sqrt(1.0+temp2);
		if (zPrime.v[0] < 0)
			cos_phi=-cos_phi;
		if (zPrime.v[1] < 0)
			sin_phi=-sin_phi;

	}
	/* ZYZ convention */

	m[0][0]=cos_theta*cos_phi;
	m[1][0]=cos_theta*sin_phi;;
	m[2][0]=-mod_sin_theta;
	m[0][1]=-sin_phi;
	m[1][1]=cos_phi;
	m[2][1]=0.0;
	m[0][2]=mod_sin_theta*cos_phi;
	m[1][2]=mod_sin_theta*sin_phi;
	m[2][2]=cos_theta;

	/* psi angle is zero */
	result.v[2]=0.0;

	/* theta angle */
	// 0 ≤ theta ≤ π
	result.v[1]=acos(cos_theta);

	/* phi angle */
	// 0 ≤ theta ≤ 2π
	result.v[0]=acos(cos_phi);
	if (sin_phi < 0.0)
		result.v[0]=2*M_PI - result.v[0];

	return result;
}

void cMatrix33::computeEulerDecompositionZY(const double &theta, double &phi) {

	// perform an Euler decomposition of a matrix assumed to be a rotation matrix
	//
	// R=R(z,psi)R(y,theta)R(z,phi)

	cVector3 x(m[0]);
	cVector3 y(m[1]);
	cVector3 z(m[2]);

	cVector3 x0(1.0,0.0,0.0);
	cVector3 y0(0.0,1.0,0.0);
	cVector3 z0(0.0,0.0,1.0);

	//first rotation
	cVector3 n=z0^z;
	n.normalize();
	std::cout<<"n="<<std::endl;
	n.print();
	std::cout<<"z="<<std::endl;
	z.print();

	cVector3 xPrime=x*cos(theta)+n*(n|x)*(1-cos(theta))+(x^n)*sin(theta);
	cVector3 yPrime=y*cos(theta)+n*(n|y)*(1-cos(theta))+(y^n)*sin(theta);
	cVector3 zPrime=z*cos(theta)+n*(n|z)*(1-cos(theta))+(z^n)*sin(theta);
	cMatrix33 RPrime(xPrime, yPrime, z0);

	std::cout<<"xPrime="<<std::endl;
	xPrime.print();
	std::cout<<"yPrime="<<std::endl;
	yPrime.print();
	std::cout<<"zPrime="<<std::endl;
	zPrime.print();
	std::cout<<"x="<<std::endl;
	x.print();
	std::cout<<"y="<<std::endl;
	y.print();
	std::cout<<"xPrime*yPrime="<< (yPrime|xPrime) <<std::endl;
	std::cout<<"xPrime*z0="<< (z0|xPrime) <<std::endl;
	std::cout<<"yPrime*z0="<< (z0|yPrime) <<std::endl;

	std::cout<<"RPrime="<<RPrime.test()<<std::endl;
	RPrime.print();
	std::cout<<"this="<<test()<<std::endl;
	print();
	cMatrix33 RPrimeInverse=RPrime.Inverse();
	std::cout<<"RPrimeInverse="<<std::endl;
	(RPrimeInverse).print();
	double trace=(RPrimeInverse).Tr();
	std::cout<<"trace= "<<trace<<std::endl;
	std::cout<<"phi= "<<acos(0.5*(trace-1.0))<<std::endl;
	phi=acos(0.5*(trace-1.0));
	//if (theta<M_PI_2) phi+=M_PI;

}

void cMatrix33::makeEulerRotationZYZ(double phi, double theta, double psi) {

	m[0][0]=-sin(psi)*sin(phi)+cos(theta)*cos(phi)*cos(psi);
	m[0][1]=sin(psi)*cos(phi)+cos(theta)*sin(phi)*cos(psi);
	m[0][2]=-cos(psi)*sin(theta);
	m[1][0]=-cos(psi)*sin(phi)-cos(theta)*cos(phi)*sin(psi);
	m[1][1]=cos(psi)*cos(phi)-cos(theta)*sin(phi)*sin(psi);
	m[1][2]=sin(psi)*sin(theta);
	m[2][0]=sin(theta)*cos(phi);
	m[2][1]=sin(theta)*sin(phi);
	m[2][2]=cos(theta);

}

void cMatrix33::makeEulerRotationXYZ(double phi, double theta, double psi) {

	m[0][0]=cos(psi)*cos(phi);
	m[1][0]=sin(psi)*cos(phi);
	m[2][0]=-sin(phi);
	m[0][1]=sin(theta)*cos(psi)*sin(phi)-cos(theta)*sin(psi);
	m[1][1]=cos(theta)*cos(psi)+sin(theta)*sin(psi)*sin(phi);
	m[2][1]=sin(theta)*cos(phi);
	m[0][2]=sin(theta)*sin(psi)+cos(theta)*cos(psi)*sin(phi);
	m[1][2]=cos(theta)*sin(psi)*sin(phi)-sin(theta)*cos(psi);
	m[2][2]=cos(theta)*cos(phi);
        
        cMatrix33 Rx(      1,0,0,
                        0,cos(phi),sin(phi),
                        0,-sin(phi),cos(phi)
                    );
        cMatrix33 Ry(   cos(theta),0,-sin(theta),
                        0,1,0,
                        sin(theta),0,cos(theta)
                    );
        cMatrix33 Rz(   cos(psi),sin(psi),0,
                        -sin(psi),cos(psi),0,
                        0,0,1
                    );
        
        *this=Rz.getTranspose()*(Ry.getTranspose()*Rx.getTranspose());

}

cMatrix33 cMatrix33::rotationXYZ(double phi, double theta, double psi)
{
	cMatrix33 Rx(   1, 0       , 0,
									0, cos(phi), -sin(phi),
									0, sin(phi), cos(phi)
							);
	cMatrix33 Ry(   cos(theta) , 0, sin(theta),
									0          , 1, 0,
									-sin(theta), 0, cos(theta)
							);
	cMatrix33 Rz(   cos(psi), -sin(psi), 0,
									sin(psi), cos(psi) , 0,
									0       , 0        , 1
							);
	return Rz*Ry*Rx;
}


void cMatrix33::makeEulerRotationYZX(double theta, double psi, double phi) {

	m[0][0]=cos(theta)*cos(psi);
	m[0][1]=-sin(psi);
	m[0][2]=sin(theta)*cos(psi);
	m[1][0]=sin(phi)*sin(theta)+cos(phi)*cos(theta)*sin(psi);
	m[1][1]=cos(phi)*cos(psi);
	m[1][2]=-cos(theta)*sin(phi)+cos(phi)*sin(theta)*sin(psi);
	m[2][0]=-cos(phi)*sin(theta)+cos(theta)*sin(phi)*sin(psi);
	m[2][1]=cos(psi)*sin(phi);
	m[2][2]=cos(phi)*cos(theta)+sin(phi)*sin(theta)*sin(psi);

}

void	cMatrix33::toArray(double v[9]) {
	v[0] = m[0][0];
	v[1] = m[0][1];
	v[2] = m[0][2];
	v[3] = m[1][0];
	v[4] = m[1][1];
	v[5] = m[1][2];
	v[6] = m[2][0];
	v[7] = m[2][1];
	v[8] = m[2][2];
}

std::ostream& operator<<(std::ostream& out, const cMatrix33& M)
{
    out << std::endl;
    out << M(0,0) << " " << M(0,1) << " " << M(0,2) << std::endl;
    out << M(1,0) << " " << M(1,1) << " " << M(1,2) << std::endl;
    out << M(2,0) << " " << M(2,1) << " " << M(2,2) << std::endl;

    return out;
}

void cMatrix33::getAxis(cVector3 &axis) {
    cVector3 eigValues;
    cMatrix33 eigVectors;
    diagonalize(eigValues, eigVectors);
    eigValues.print();
}

void cMatrix33::makeUniformRotation(cVector3 &uniformVector){

	float u1 = uniformVector.v[0];
	float u2 = uniformVector.v[1];
	float u3 = uniformVector.v[2];
	float q[4];
	q[0] = sqrt(1-u1) * sin(2.0*M_PI*u2);
	q[1] = sqrt(1-u1) * cos(2.0*M_PI*u2);
	q[2] = sqrt(u1) * sin(2.0*M_PI*u3);
	q[3] = sqrt(u1) * cos(2.0*M_PI*u3);
	m[0][0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
	m[0][1] = 2.0*(q[1]*q[2] - q[0]*q[3]);
	m[0][2] = 2.0*(q[1]*q[3] + q[0]*q[2]);

	m[1][0] = 2.0*(q[1]*q[2] + q[0]*q[3]);
	m[1][1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
	m[1][2] = 2.0*(q[2]*q[3] - q[0]*q[1]);

	m[2][0] = 2.0*(q[1]*q[3] - q[0]*q[2]);
	m[2][1] = 2.0*(q[2]*q[3] + q[0]*q[1]);
	m[2][2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
}