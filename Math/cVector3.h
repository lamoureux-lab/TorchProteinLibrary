#pragma once

#include <math.h>
#include <iostream>

#include <sstream>
#include <string>
#include <fstream>

class cVector3 {
private:
    bool external;
public:

	double				*v;
    

	cVector3();
    cVector3(const cVector3& u);
    cVector3(double *v);
    ~cVector3();
	cVector3(double x);
	cVector3(double x, double y, double z);
	       
	double				operator|(const cVector3& u) const;
	cVector3			operator^(const cVector3& u) const;
	cVector3			operator+(const cVector3 &u) const;
	cVector3&			operator+=(const cVector3 &u);
	cVector3			operator-(const cVector3 &u) const;
	cVector3&			operator-=(const cVector3 &u);
	cVector3			operator*(double d) const;
	cVector3			operator*(const cVector3 &u) const;
	cVector3&			operator*=(double d);
	cVector3			operator/(double d) const;
    cVector3			operator/(const cVector3 &u) const;
	cVector3&			operator/=(double d);
	cVector3			operator-();
	bool				operator==(const cVector3& u) const;
	bool				operator!=(const cVector3& u) const;
	bool				operator<(const cVector3& u) const;
	void				operator=(const double &d) {v[0]=v[1]=v[2]=d;}
	void				operator=(const cVector3& u);
	double&				operator[](int ind) {return v[ind];};
	const double		operator[](int ind) const {return v[ind];};

    
    void				setZero();
	double				norm() const;
	double				norm2()	const;
    
};

std::ostream& operator << (std::ostream& os, cVector3 v);

typedef cVector3 *pcVector3;

cVector3 operator*(double d, cVector3 u);

