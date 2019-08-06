#include <math.h>
#include <iostream>

#include "cVector3.h"

template <typename T> cVector3<T>::cVector3() {
    external = false;
    v = new T[3];
}

template <typename T> cVector3<T>::cVector3(T *v){
    external = true;
    this->v = v;
}

template <typename T> cVector3<T>::cVector3(T x){ 
    external = false;
    v = new T[3];
    v[0]=x;v[1]=x;v[2]=x;
}

template <typename T> cVector3<T>::cVector3(T x, T y, T z){ 
    external = false;
    v = new T[3];
    v[0]=x;v[1]=y;v[2]=z;
}

template <typename T> cVector3<T>::~cVector3(){
    
    if(!external){
        delete [] v;
    }else{
    }
}

template <typename T> cVector3<T>::cVector3(const cVector3& u){
    this->external = u.external;
    if(u.external)
        this->v = u.v;
    else{
        v = new T[3];
        v[0]=u.v[0];v[1]=u.v[1];v[2]=u.v[2];
    }
}

template <typename T> void cVector3<T>::setZero() { v[0]=v[1]=v[2]=0.0; }
template <typename T> T cVector3<T>::norm() const { return (T)sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
template <typename T> T cVector3<T>::norm2() const { return v[0]*v[0]+v[1]*v[1]+v[2]*v[2]; }
template <typename T> T cVector3<T>::operator|(const cVector3<T> &u) const { return v[0]*u.v[0]+v[1]*u.v[1]+v[2]*u.v[2]; }
template <typename T> cVector3<T> cVector3<T>::operator^(const cVector3<T> &u) const { return cVector3<T>(v[1]*u.v[2]-v[2]*u.v[1],v[2]*u.v[0]-v[0]*u.v[2],v[0]*u.v[1]-v[1]*u.v[0]); }
template <typename T> cVector3<T> cVector3<T>::operator+(const cVector3<T> &u) const { return cVector3<T>(v[0]+u.v[0],v[1]+u.v[1],v[2]+u.v[2]); }
template <typename T> cVector3<T>& cVector3<T>::operator+=(const cVector3<T> &u) { v[0]+=u.v[0];v[1]+=u.v[1];v[2]+=u.v[2];return *this; }
template <typename T> cVector3<T> cVector3<T>::operator-(const cVector3<T> &u) const { return cVector3<T>(v[0]-u.v[0],v[1]-u.v[1],v[2]-u.v[2]); }
template <typename T> cVector3<T>& cVector3<T>::operator-=(const cVector3<T> &u) { v[0]-=u.v[0];v[1]-=u.v[1];v[2]-=u.v[2];return *this; }
template <typename T> cVector3<T> cVector3<T>::operator*(T d) const { return cVector3<T>(v[0]*d,v[1]*d,v[2]*d); }
template <typename T> cVector3<T>& cVector3<T>::operator*=(T d) { v[0]*=d;v[1]*=d;v[2]*=d;return *this; }
template <typename T> cVector3<T> cVector3<T>::operator/(T d) const { return cVector3<T>(v[0]/d,v[1]/d,v[2]/d); }
template <typename T> cVector3<T> cVector3<T>::operator/(const cVector3<T> &u) const { return cVector3<T>(v[0]/u[0],v[1]/u[1],v[2]/u[2]); }
template <typename T> cVector3<T>& cVector3<T>::operator/=(T d) { v[0]/=d;v[1]/=d;v[2]/=d;return *this; }
template <typename T> cVector3<T> cVector3<T>::operator-() { return cVector3<T>(-v[0],-v[1],-v[2]); }
template <typename T> bool cVector3<T>::operator==(const cVector3<T> &u) const { return (v[0]==u.v[0])&&(v[1]==u.v[1])&&(v[2]==u.v[2]); }
template <typename T> bool cVector3<T>::operator!=(const cVector3<T> &u) const { return (v[0]!=u.v[0])||(v[1]!=u.v[1])||(v[2]!=u.v[2]); }
template <typename T> bool cVector3<T>::operator<(const cVector3<T> &u) const { if (v[0]>=u.v[0]) return false;if (v[1]>=u.v[1]) return false;if (v[2]>=u.v[2]) return false;return true; }
template <typename T> cVector3<T>& cVector3<T>::operator=(const cVector3<T> &u){
    v[0]=u.v[0];v[1]=u.v[1];v[2]=u.v[2];
    return *this;
}

template <typename T> cVector3<T> cVector3<T>::operator*(const cVector3<T> &u) const { return cVector3<T>(v[0]*u.v[0],v[1]*u.v[1],v[2]*u.v[2]); }

template <typename T> cVector3<T> operator*(T d, cVector3<T> u) {

	return cVector3<T>(d*u.v[0],d*u.v[1],d*u.v[2]);

}

template <typename T> std::ostream& operator << (std::ostream& os, cVector3<T> v)
{

    os<<v.v[0]<<" "<<v.v[1]<<" "<<v.v[2];
    return os;
}

template class cVector3<float>;
template std::ostream& operator << (std::ostream& os, cVector3<float> v);
template cVector3<float> operator*(float d, cVector3<float> u);

template class cVector3<double>;
template std::ostream& operator << (std::ostream&, cVector3<double>);
template cVector3<double> operator*(double, cVector3<double>);
