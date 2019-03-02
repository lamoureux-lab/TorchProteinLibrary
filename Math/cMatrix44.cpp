#include <cMatrix44.h>
#include <math.h>


cMatrix44::cMatrix44(){
	external = false;
    m = new double[16];
	// for(int i=0;i<4;i++)for(int j=0;j<4;j++)(*this)(i,j)=0.0;
}
cMatrix44::cMatrix44(double mat[4][4]){
	external = false;
    m = new double[16];
	for(int i=0;i<4;i++)for(int j=0;j<4;j++)(*this)(i,j)=mat[i][j];
}
cMatrix44::cMatrix44(double *mat){
	external = true;
	m = mat;
}
cMatrix44::cMatrix44(const cMatrix33 &rot, const cVector3 &shift){
	external = false;
    m = new double[16];
	for(int i=0;i<3;i++){
		(*this)(3,i)=shift[i];
		for(int j=0;j<3;j++)
			(*this)(i,j)=rot(i,j);
	}
	(*this)(4,4)=1;
}

cMatrix44::~cMatrix44(){
	if(external){
        return;
    }else{
		delete [] m;
    }
}


void cMatrix44::setDihedral(const double phi, const double psi, const double R){
	m[0]=cos(psi);            	m[1]=sin(phi)*sin(psi);	m[2]=cos(phi)*sin(psi);		m[3]=R*cos(psi);
	m[4]=0.0;					m[5]=cos(phi); 			m[6]=-sin(phi); 			m[7]=0.0;
	m[8]=-sin(psi);   			m[9]=sin(phi)*cos(psi);	m[10]=cos(phi)*cos(psi); 	m[11]=-R*sin(psi);
	m[12]=0.0;				    m[13]=0.0;				m[14]=0.0;		 			m[15]=1.0;
}

void cMatrix44::setDihedralDphi(const double phi, const double psi, const double R){
	m[0]=0; 	m[1]=cos(phi)*sin(psi);		m[2]=-sin(phi)*sin(psi);	m[3]=0;
	m[4]=0;		m[5]=-sin(phi); 			m[6]=-cos(phi);				m[7]=0;
	m[8]=0;  	m[9]=cos(phi)*cos(psi);		m[10]=-sin(phi)*cos(psi);	m[11]=0;
	m[12]=0;	m[13]=0.0;					m[14]=0.0;					m[15]=0;
}

void cMatrix44::setRx(const double angle){
	m[0]=1;    	m[1]=0;	        	m[2]=0;            m[3]=0;
	m[4]=0;		m[5]=cos(angle);	m[6]=-sin(angle);  m[7]=0;
	m[8]=0;    	m[9]=sin(angle);   	m[10]=cos(angle);  m[11]=0;
	m[12]=0;	m[13]=0;	        m[14]=0;           m[15]=1;
}
void cMatrix44::setRy(const double angle){
	m[0]=cos(angle); 	m[1]=0;		m[2]=sin(angle);	m[3]=0;
	m[4]=0;	        	m[5]=1;		m[6]=0;	        	m[7]=0;
	m[8]=-sin(angle); 	m[9]=0;    	m[10]=cos(angle);   m[11]=0;
	m[12]=0;			m[13]=0;	m[14]=0;	        m[15]=1;
}
void cMatrix44::setRz(const double angle){
	m[0]=cos(angle); 	m[1]=-sin(angle);	m[2]=0;		m[3]=0;	
	m[4]=sin(angle);	m[5]=cos(angle);	m[6]=0;		m[7]=0;
	m[8]=0;            	m[9]=0;	        	m[10]=1;    m[11]=0;
	m[12]=0;			m[13]=0;			m[14]=0;	m[15]=1;
}
void cMatrix44::setDRx(const double angle){
	m[0]=0;    	m[1]=0;	        	m[2]=0;            m[3]=0;
	m[4]=0;		m[5]=-sin(angle);	m[6]=-cos(angle);  m[7]=0;
	m[8]=0;    	m[9]=cos(angle);   	m[10]=-sin(angle); m[11]=0;
	m[12]=0;	m[13]=0;	        m[14]=0;           m[15]=0;
}

void cMatrix44::setT(const double R, const char axis){
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



cMatrix44 cMatrix44::operator*(const cMatrix44 &mat) const{
	cMatrix44 res;
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++){
			res(i,j)=0.0;
			for(int k=0;k<4;k++)
				res(i,j)+=(*this)(i,k)*mat(k,j);
		}
	
	return res;
}

cVector3 cMatrix44::operator*(const cVector3 &vec) const{
	double vec4[4], vec4_new[4];
	vec4[0]=vec[0];vec4[1]=vec[1];vec4[2]=vec[2];vec4[3]=1.0;

	for(int i=0;i<4;i++){
		vec4_new[i]=0;
		for(int j=0;j<4;j++)
			vec4_new[i]+=(*this)(i,j)*vec4[j];
	}

	cVector3 vec3(vec4_new[0], vec4_new[1], vec4_new[2]);
	return vec3;
}
void cMatrix44::setIdentity(){
	for (int i=0;i<4;i++) {
		for (int j=0;j<4;j++) {
			if(i==j)(*this)(i,j)=1.0;
			else (*this)(i,j)=0.0;
	}}
			
}

cMatrix44 invertTransform44(const cMatrix44 &mat){
	
	cMatrix33 invRot(mat(0,0), mat(1,0), mat(2,0), mat(0,1), mat(1,1), mat(2,1), mat(0,2), mat(1,2), mat(2,2));
	cVector3 trans(mat(0,3), mat(1,3), mat(2,3));
	cVector3 invTrans = invRot*trans*(-1);
	double m[4][4];
	m[0][0] = invRot.m[0][0]; m[0][1] = invRot.m[0][1]; m[0][2] = invRot.m[0][2]; m[0][3] = invTrans.v[0];
	m[1][0] = invRot.m[1][0]; m[1][1] = invRot.m[1][1]; m[1][2] = invRot.m[1][2]; m[1][3] = invTrans.v[1];
	m[2][0] = invRot.m[2][0]; m[2][1] = invRot.m[2][1]; m[2][2] = invRot.m[2][2]; m[2][3] = invTrans.v[2];
	m[3][0] = 0; m[3][1] = 0; m[3][2] = 0; m[3][3] = 1;
	return cMatrix44(m);
}

void cMatrix44::operator=(const cMatrix44& u){
	for(int i=0; i<16; i++)
		m[i]=u.m[i];
}

void cMatrix44::print()	{

	double SMALL_VALUE=10e-200;
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
