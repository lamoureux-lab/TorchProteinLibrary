#include <cMatrix44.h>
#include <math.h>


template <typename T> cMatrix44<T>::cMatrix44(){
	external = false;
    this->m = new T[16];
}
template <typename T> cMatrix44<T>::cMatrix44(const cMatrix44<T>& other){
	this->external = other.external;
	if(other.external){
		this->m = other.m;
	}else{
		this->m = new T[16];
		for(int i=0;i<4;i++)for(int j=0;j<4;j++)(*this)(i,j)=other(i,j);
	}
}
template <typename T> cMatrix44<T>::cMatrix44(T mat[4][4]){
	external = false;
    this->m = new T[16];
	for(int i=0;i<4;i++)for(int j=0;j<4;j++)(*this)(i,j)=mat[i][j];
}
template <typename T> cMatrix44<T>::cMatrix44(T *mat){
	external = true;
	this->m = mat;
}
template <typename T> cMatrix44<T>::cMatrix44(const cMatrix33<T> &rot, const cVector3<T> &shift){
	external = false;
    this->m = new T[16];
	for(int i=0;i<3;i++){
		(*this)(3,i)=shift[i];
		for(int j=0;j<3;j++)
			(*this)(i,j)=rot(i,j);
	}
	(*this)(4,4)=1;
}

template <typename T> cMatrix44<T>::~cMatrix44(){
	if(external){
        return;
    }else{
		delete [] m;
    }
}


template <typename T> void cMatrix44<T>::setDihedral(const T phi, const T psi, const T R){
	m[0]=cos(psi);            	m[1]=sin(phi)*sin(psi);	m[2]=cos(phi)*sin(psi);		m[3]=R*cos(psi);
	m[4]=0.0;					m[5]=cos(phi); 			m[6]=-sin(phi); 			m[7]=0.0;
	m[8]=-sin(psi);   			m[9]=sin(phi)*cos(psi);	m[10]=cos(phi)*cos(psi); 	m[11]=-R*sin(psi);
	m[12]=0.0;				    m[13]=0.0;				m[14]=0.0;		 			m[15]=1.0;
}

template <typename T> void cMatrix44<T>::setDihedralDphi(const T phi, const T psi, const T R){
	m[0]=0; 	m[1]=cos(phi)*sin(psi);		m[2]=-sin(phi)*sin(psi);	m[3]=0;
	m[4]=0;		m[5]=-sin(phi); 			m[6]=-cos(phi);				m[7]=0;
	m[8]=0;  	m[9]=cos(phi)*cos(psi);		m[10]=-sin(phi)*cos(psi);	m[11]=0;
	m[12]=0;	m[13]=0.0;					m[14]=0.0;					m[15]=0;
}

template <typename T> void cMatrix44<T>::setRx(const T angle){
	m[0]=1;    	m[1]=0;	        	m[2]=0;            m[3]=0;
	m[4]=0;		m[5]=cos(angle);	m[6]=-sin(angle);  m[7]=0;
	m[8]=0;    	m[9]=sin(angle);   	m[10]=cos(angle);  m[11]=0;
	m[12]=0;	m[13]=0;	        m[14]=0;           m[15]=1;
}
template <typename T> void cMatrix44<T>::setRy(const T angle){
	m[0]=cos(angle); 	m[1]=0;		m[2]=sin(angle);	m[3]=0;
	m[4]=0;	        	m[5]=1;		m[6]=0;	        	m[7]=0;
	m[8]=-sin(angle); 	m[9]=0;    	m[10]=cos(angle);   m[11]=0;
	m[12]=0;			m[13]=0;	m[14]=0;	        m[15]=1;
}
template <typename T> void cMatrix44<T>::setRz(const T angle){
	m[0]=cos(angle); 	m[1]=-sin(angle);	m[2]=0;		m[3]=0;	
	m[4]=sin(angle);	m[5]=cos(angle);	m[6]=0;		m[7]=0;
	m[8]=0;            	m[9]=0;	        	m[10]=1;    m[11]=0;
	m[12]=0;			m[13]=0;			m[14]=0;	m[15]=1;
}
template <typename T> void cMatrix44<T>::setDRx(const T angle){
	m[0]=0;    	m[1]=0;	        	m[2]=0;            m[3]=0;
	m[4]=0;		m[5]=-sin(angle);	m[6]=-cos(angle);  m[7]=0;
	m[8]=0;    	m[9]=cos(angle);   	m[10]=-sin(angle); m[11]=0;
	m[12]=0;	m[13]=0;	        m[14]=0;           m[15]=0;
}

template <typename T> void cMatrix44<T>::setT(const T R, const char axis){
 	int ax_ind;
    switch(axis){
        case 'x':
            ax_ind=0;
            break;
        case 'y':
            ax_ind=1;
            break;
        case 'z':
            ax_ind=2;
            break;
        default:
            throw(std::string("cMatrix44::getT Axis selection error"));
            break;
    };
    this->setIdentity();
    (*this)(ax_ind, 3) = R;
}



template <typename T> cMatrix44<T> cMatrix44<T>::operator*(const cMatrix44<T> &mat) const{
	cMatrix44<T> res;
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++){
			res(i,j)=0.0;
			for(int k=0;k<4;k++)
				res(i,j)+=(*this)(i,k)*mat(k,j);
		}
	
	return res;
}

template <typename T> cVector3<T> cMatrix44<T>::operator*(const cVector3<T> &vec) const{
	T vec4[4], vec4_new[4];
	vec4[0]=vec[0];vec4[1]=vec[1];vec4[2]=vec[2];vec4[3]=1.0;

	for(int i=0;i<4;i++){
		vec4_new[i]=0;
		for(int j=0;j<4;j++)
			vec4_new[i]+=(*this)(i,j)*vec4[j];
	}

	cVector3<T> vec3(vec4_new[0], vec4_new[1], vec4_new[2]);
	return vec3;
}
template <typename T> void cMatrix44<T>::setIdentity(){
	for (int i=0;i<4;i++) {
		for (int j=0;j<4;j++) {
			if(i==j)(*this)(i,j)=1.0;
			else (*this)(i,j)=0.0;
	}}
			
}

template <typename T> cMatrix44<T> invertTransform44(const cMatrix44<T> &mat){
	
	cMatrix33<T> invRot(mat(0,0), mat(1,0), mat(2,0), mat(0,1), mat(1,1), mat(2,1), mat(0,2), mat(1,2), mat(2,2));
	cVector3<T> trans(mat(0,3), mat(1,3), mat(2,3));
	cVector3<T> invTrans = invRot*trans*(-1);
	T m[4][4];
	m[0][0] = invRot.m[0][0]; m[0][1] = invRot.m[0][1]; m[0][2] = invRot.m[0][2]; m[0][3] = invTrans.v[0];
	m[1][0] = invRot.m[1][0]; m[1][1] = invRot.m[1][1]; m[1][2] = invRot.m[1][2]; m[1][3] = invTrans.v[1];
	m[2][0] = invRot.m[2][0]; m[2][1] = invRot.m[2][1]; m[2][2] = invRot.m[2][2]; m[2][3] = invTrans.v[2];
	m[3][0] = 0; m[3][1] = 0; m[3][2] = 0; m[3][3] = 1;
	return cMatrix44<T>(m);
}

template <typename T> void cMatrix44<T>::operator=(const cMatrix44<T>& u){
	for(int i=0; i<16; i++)
		m[i]=u.m[i];
}

template <typename T> void cMatrix44<T>::print()	{

	T SMALL_VALUE=10e-200;
	for (int i=0;i<4;i++) {
		for (int j=0;j<4;j++) {
			if (fabs((*this)(i,j)) < SMALL_VALUE && fabs((*this)(i,j)) > 0.0)
				std::cout << "\t[" << "O" << "]";
			else {
				std::cout << "\t[" << (*this)(i,j) << "]";
			}
		}
		std::cout << std::endl;

	}

	std::cout << std::endl;
	std::cout << std::endl;
}

template class cMatrix44<float>;
template cMatrix44<float> invertTransform44(const cMatrix44<float>&);

template class cMatrix44<double>;
template cMatrix44<double> invertTransform44(const cMatrix44<double>&);