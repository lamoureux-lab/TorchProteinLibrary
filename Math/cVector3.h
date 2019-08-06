#pragma once

#include <math.h>
#include <iostream>

#include <sstream>
#include <string>
#include <fstream>

template <typename T>
class cVector3 {
private:
    bool external;
public:

	T				*v;
    

	cVector3();
    cVector3(const cVector3<T>& u);
    cVector3(T *v);
    ~cVector3();
	cVector3(T x);
	cVector3(T x, T y, T z);
	       
	T					operator|(const cVector3<T> &u) const;
	cVector3<T>			operator^(const cVector3<T> &u) const;
	cVector3<T>			operator+(const cVector3<T> &u) const;
	cVector3<T>&		operator+=(const cVector3<T> &u);
	cVector3<T>			operator-(const cVector3<T> &u) const;
	cVector3<T>&		operator-=(const cVector3<T> &u);
	cVector3<T>			operator*(T d) const;
	cVector3<T>			operator*(const cVector3<T> &u) const;
	cVector3<T>&		operator*=(T d);
	cVector3<T>			operator/(T d) const;
    cVector3<T>			operator/(const cVector3<T> &u) const;
	cVector3<T>&		operator/=(T d);
	cVector3<T>			operator-();
	bool				operator==(const cVector3<T> &u) const;
	bool				operator!=(const cVector3<T> &u) const;
	bool				operator<(const cVector3<T> &u) const;
	cVector3<T>&		operator=(const T &d) {v[0]=v[1]=v[2]=d; return *this;}
	cVector3<T>&		operator=(cVector3<T> const &u);
	T&					operator[](int ind) {return v[ind];};
	const T				operator[](int ind) const {return v[ind];};

    
    void				setZero();
	T					norm() const;
	T					norm2()	const;
    
};
template <typename T> std::ostream& operator << (std::ostream& os, cVector3<T> v);
template <typename T> cVector3<T> operator*(T d, cVector3<T> u);