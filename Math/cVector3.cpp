#include <math.h>
#include <iostream>

#include "cVector3.h"

cVector3::cVector3() {
    external = false;
    v = new double[3];
}
cVector3::cVector3(double *v){
    external = true;
    this->v = v;
}
cVector3::cVector3(double x){ 
    external = false;
    v = new double[3];
    v[0]=x;v[1]=x;v[2]=x;
}
cVector3::cVector3(double x, double y, double z){ 
    external = false;
    v = new double[3];
    v[0]=x;v[1]=y;v[2]=z;
}
cVector3::~cVector3(){
    
    if(!external){
        delete [] v;
    }else{
    }
}

cVector3::cVector3(const cVector3& u){
    this->external = u.external;
    if(u.external)
        this->v = u.v;
    else{
        v = new double[3];
        v[0]=u.v[0];v[1]=u.v[1];v[2]=u.v[2];
    }
}


void cVector3::setZero() { v[0]=v[1]=v[2]=0.0; }
double cVector3::norm() const { return (double)sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
double cVector3::norm2() const { return v[0]*v[0]+v[1]*v[1]+v[2]*v[2]; }
double cVector3::operator|(const cVector3& u) const { return v[0]*u.v[0]+v[1]*u.v[1]+v[2]*u.v[2]; }
cVector3 cVector3::operator^(const cVector3& u) const { return cVector3(v[1]*u.v[2]-v[2]*u.v[1],v[2]*u.v[0]-v[0]*u.v[2],v[0]*u.v[1]-v[1]*u.v[0]); }
cVector3 cVector3::operator+(const cVector3 &u) const { return cVector3(v[0]+u.v[0],v[1]+u.v[1],v[2]+u.v[2]); }
cVector3& cVector3::operator+=(const cVector3 &u) { v[0]+=u.v[0];v[1]+=u.v[1];v[2]+=u.v[2];return *this; }
cVector3 cVector3::operator-(const cVector3 &u) const { return cVector3(v[0]-u.v[0],v[1]-u.v[1],v[2]-u.v[2]); }
cVector3& cVector3::operator-=(const cVector3 &u) { v[0]-=u.v[0];v[1]-=u.v[1];v[2]-=u.v[2];return *this; }
cVector3 cVector3::operator*(double d) const { return cVector3(v[0]*d,v[1]*d,v[2]*d); }
cVector3& cVector3::operator*=(double d) { v[0]*=d;v[1]*=d;v[2]*=d;return *this; }
cVector3 cVector3::operator/(double d) const { return cVector3(v[0]/d,v[1]/d,v[2]/d); }
cVector3 cVector3::operator/(const cVector3 &u) const { return cVector3(v[0]/u[0],v[1]/u[1],v[2]/u[2]); }
cVector3& cVector3::operator/=(double d) { v[0]/=d;v[1]/=d;v[2]/=d;return *this; }
cVector3 cVector3::operator-() { return cVector3(-v[0],-v[1],-v[2]); }
bool cVector3::operator==(const cVector3& u) const { return (v[0]==u.v[0])&&(v[1]==u.v[1])&&(v[2]==u.v[2]); }
bool cVector3::operator!=(const cVector3& u) const { return (v[0]!=u.v[0])||(v[1]!=u.v[1])||(v[2]!=u.v[2]); }
bool cVector3::operator<(const cVector3& u) const { if (v[0]>=u.v[0]) return false;if (v[1]>=u.v[1]) return false;if (v[2]>=u.v[2]) return false;return true; }
void cVector3::operator=(const cVector3& u){
    v[0]=u.v[0];v[1]=u.v[1];v[2]=u.v[2];
}

cVector3 operator*(double d, cVector3 u) {

	return cVector3(d*u.v[0],d*u.v[1],d*u.v[2]);

}

cVector3 cVector3::operator*(const cVector3 &u) const { return cVector3(v[0]*u.v[0],v[1]*u.v[1],v[2]*u.v[2]); }

std::ostream& operator << (std::ostream& os, cVector3 v)
{

    os<<v.v[0]<<" "<<v.v[1]<<" "<<v.v[2];
    return os;
}