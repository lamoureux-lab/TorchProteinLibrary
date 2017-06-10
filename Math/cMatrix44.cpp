#include <cMatrix44.h>
#include <math.h>


cMatrix44::cMatrix44(){
	for(int i=0;i<4;i++)for(int j=0;j<4;j++)(*this)(i,j)=0.0;
	
}
cMatrix44::cMatrix44(double mat[4][4]){
	for(int i=0;i<4;i++){
		for(int j=0;j<4;j++){
			m[i][j]=mat[i][j];
	}}
}
cMatrix44::cMatrix44(const cMatrix33 &rot, const cVector3 &shift){
	for(int i=0;i<3;i++){
		(*this)(3,i)=shift[i];
		for(int j=0;j<3;j++)
			(*this)(i,j)=rot(i,j);
	}
	(*this)(4,4)=1;
	
}

cMatrix44::cMatrix44(const float alpha, const float beta, const float R){
	m[0][0]=cos(alpha);            	m[0][1]=-sin(alpha)*cos(beta);  m[0][2]=sin(alpha)*sin(beta);		m[0][3]=-R*sin(alpha)*cos(beta);
	m[1][0]=sin(alpha);				m[1][1]=cos(alpha)*cos(beta); 	m[1][2]=-cos(alpha)*sin(beta); 		m[1][3]=R*cos(alpha)*cos(beta);
	m[2][0]=0.0;   					m[2][1]=sin(beta);				m[2][2]=cos(beta); 					m[2][3]=R*sin(beta);
	m[3][0]=0.0;				    m[3][1]=0.0;					m[3][2]=0.0;		 				m[3][3]=1.0;
}

void cMatrix44::setGradAlpha(const float alpha, const float beta, const float R){
	m[0][0]=-sin(alpha);            m[0][1]=-cos(alpha)*cos(beta);  m[0][2]=cos(alpha)*sin(beta);		m[0][3]=-R*cos(alpha)*cos(beta);
	m[1][0]=cos(alpha);  			m[1][1]=-sin(alpha)*cos(beta); 	m[1][2]=sin(alpha)*sin(beta);		m[1][3]=-R*sin(alpha)*cos(beta);
	m[2][0]=0.0;   					m[2][1]=0.0;				 	m[2][2]=0.0; 						m[2][3]=0.0;
	m[3][0]=0.0;				    m[3][1]=0.0;					m[3][2]=0.0;		 				m[3][3]=0.0;
}
void cMatrix44::setGradBeta(const float alpha, const float beta, const float R){
	m[0][0]=0.0;		            m[0][1]=sin(alpha)*sin(beta);	m[0][2]=sin(alpha)*cos(beta);		m[0][3]=R*sin(alpha)*sin(beta);
	m[1][0]=0.0;					m[1][1]=-cos(alpha)*sin(beta); 	m[1][2]=-cos(alpha)*cos(beta); 		m[1][3]=-R*cos(alpha)*sin(beta);
	m[2][0]=0.0;   					m[2][1]=cos(beta); 				m[2][2]=-sin(beta); 				m[2][3]=R*cos(beta);
	m[3][0]=0.0;				    m[3][1]=0.0;					m[3][2]=0.0;		 				m[3][3]=0.0;
}


cMatrix44 cMatrix44::operator*(const cMatrix44 &mat) const{
	double res[4][4];
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++){
			res[i][j]=0.0;
			for(int k=0;k<4;k++)
				res[i][j]+=m[i][k]*mat.m[k][j];
		}
	cMatrix44 Res(res);
	return Res;
}

cVector3 cMatrix44::operator*(const cVector3 &vec) const{
	double vec4[4], vec4_new[4];
	vec4[0]=vec[0];vec4[1]=vec[1];vec4[2]=vec[2];vec4[3]=1.0;

	for(int i=0;i<4;i++){
		vec4_new[i]=0;
		for(int j=0;j<4;j++)
			vec4_new[i]+=m[i][j]*vec4[j];
	}

	cVector3 vec3(vec4_new[0], vec4_new[1], vec4_new[2]);
	return vec3;
}
void cMatrix44::ones(){
	for (int i=0;i<4;i++) {
		for (int j=0;j<4;j++) {
			if(i==j)m[i][j]=1.0;
			else m[i][j]=0.0;
	}}
			
}
void cMatrix44::print()	{

	double SMALL_VALUE=10e-200;
	for (int i=0;i<4;i++) {
		for (int j=0;j<4;j++) {
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