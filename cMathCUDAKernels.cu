#include "cMathCUDAKernels.h"

#define KAPPA1 (3.14159 - 1.9391)
#define KAPPA2 (3.14159 - 2.061)
#define KAPPA3 (3.14159 -2.1186)
#define OMEGACIS -3.1318

#define R_CA_C 1.525
#define R_C_N 1.330
#define R_N_CA 1.460

#define CA_C_N (M_PI - 2.1186)
#define C_N_CA (M_PI - 1.9391)
#define N_CA_C  (M_PI - 2.061)


__device__ void getRotationMatrix(REAL *d_data, REAL alpha, REAL beta, REAL R){
	d_data[0]=cos(alpha);   d_data[1]=-sin(alpha)*cos(beta);d_data[2]=sin(alpha)*sin(beta);	d_data[3]=-R*sin(alpha)*cos(beta);
	d_data[4]=sin(alpha);	d_data[5]=cos(alpha)*cos(beta); d_data[6]=-cos(alpha)*sin(beta);d_data[7]=R*cos(alpha)*cos(beta);
	d_data[8]=0.0;   		d_data[9]=sin(beta);			d_data[10]=cos(beta); 			d_data[11]=R*sin(beta);
	d_data[12]=0.0;			d_data[13]=0.0;					d_data[14]=0.0;		 			d_data[15]=1.0;
}

__device__ void get33RotationMatrix(REAL *d_data, REAL alpha, REAL beta){
    d_data[0]=cos(alpha);   d_data[1]=-sin(alpha)*cos(beta);d_data[2]=sin(alpha)*sin(beta);	
	d_data[3]=sin(alpha);	d_data[4]=cos(alpha)*cos(beta); d_data[5]=-cos(alpha)*sin(beta);
	d_data[6]=0.0;   		d_data[7]=sin(beta);			d_data[8]=cos(beta); 			
}

__device__ void getRotationMatrixDAlpha(REAL *d_data, REAL alpha, REAL beta, REAL R){
	d_data[0]=-sin(alpha);  d_data[1]=-cos(alpha)*cos(beta);    d_data[2]=cos(alpha)*sin(beta);		d_data[3]=-R*cos(alpha)*cos(beta);
	d_data[4]=cos(alpha);   d_data[5]=-sin(alpha)*cos(beta); 	d_data[6]=sin(alpha)*sin(beta);		d_data[7]=-R*sin(alpha)*cos(beta);
	d_data[8]=0.0;   		d_data[9]=0.0;				 	    d_data[10]=0.0; 					d_data[11]=0.0;
	d_data[12]=0.0;			d_data[13]=0.0;					    d_data[14]=0.0;		 				d_data[15]=0.0;
}

__device__ void get33RotationMatrixDAlpha(REAL *d_data, REAL alpha, REAL beta){
	d_data[0]=-sin(alpha);  d_data[1]=-cos(alpha)*cos(beta);    d_data[2]=cos(alpha)*sin(beta);
	d_data[3]=cos(alpha);   d_data[4]=-sin(alpha)*cos(beta); 	d_data[5]=sin(alpha)*sin(beta);
	d_data[6]=0.0;   		d_data[7]=0.0;				 	    d_data[8]=0.0; 			
}

__device__ void getRotationMatrixDBeta(REAL *d_data, REAL alpha, REAL beta, REAL R){
	d_data[0]=0.0;          d_data[1]=sin(alpha)*sin(beta);	d_data[2]=sin(alpha)*cos(beta);		d_data[3]=R*sin(alpha)*sin(beta);
	d_data[4]=0.0;			d_data[5]=-cos(alpha)*sin(beta); 	d_data[6]=-cos(alpha)*cos(beta);    d_data[7]=-R*cos(alpha)*sin(beta);
	d_data[8]=0.0;   		d_data[9]=cos(beta); 				d_data[10]=-sin(beta); 				d_data[11]=R*cos(beta);
	d_data[12]=0.0;			d_data[13]=0.0;					    d_data[14]=0.0;		 				d_data[15]=0.0;
}

__device__ void get33RotationMatrixDBeta(REAL *d_data, REAL alpha, REAL beta){
	d_data[0]=0.0;          d_data[1]=sin(alpha)*sin(beta);	    d_data[2]=sin(alpha)*cos(beta);	
	d_data[3]=0.0;			d_data[4]=-cos(alpha)*sin(beta); 	d_data[5]=-cos(alpha)*cos(beta);
	d_data[6]=0.0;   		d_data[7]=cos(beta); 				d_data[8]=-sin(beta); 			
}

__device__ void getRotationMatrixDihedral(REAL *d_data, REAL a, REAL b, REAL R){
	d_data[0]=cos(b); 	d_data[1]=sin(a)*sin(b);	d_data[2]=cos(a)*sin(b);	d_data[3]=R*cos(b);
	d_data[4]=0;		d_data[5]=cos(a); 			d_data[6]=-sin(a);			d_data[7]=0;
	d_data[8]=-sin(b);  d_data[9]=sin(a)*cos(b);	d_data[10]=cos(a)*cos(b);	d_data[11]=-R*sin(b);
	d_data[12]=0.0;		d_data[13]=0.0;				d_data[14]=0.0;				d_data[15]=1.0;
}
__device__ void getRotationMatrixDihedralDPsi(REAL *d_data, REAL a, REAL b, REAL R){
	d_data[0]=0; 		d_data[1]=cos(a)*sin(b);	d_data[2]=-sin(a)*sin(b);	d_data[3]=0;
	d_data[4]=0;		d_data[5]=-sin(a); 			d_data[6]=-cos(a);			d_data[7]=0;
	d_data[8]=0;  		d_data[9]=cos(a)*cos(b);	d_data[10]=-sin(a)*cos(b);	d_data[11]=0;
	d_data[12]=0;		d_data[13]=0.0;				d_data[14]=0.0;				d_data[15]=0;
}

__device__ void getRotationMatrixCalpha(REAL *d_data, REAL phi, REAL psi, bool first){
	// getRotationMatrixDihedral(d_data, 0.0, psi);
	REAL A[16], B[16], C[16], D[16];
	if(first){
		getRotationMatrixDihedral(d_data, phi, C_N_CA, R_N_CA);
	}else{
		getRotationMatrixDihedral(B, psi, N_CA_C, R_CA_C);
		getRotationMatrixDihedral(C, OMEGACIS, CA_C_N, R_C_N);
		getRotationMatrixDihedral(A, phi, C_N_CA, R_N_CA);	
		mat44Mul(B, C, D);
		mat44Mul(D, A, d_data);	
	}
}

__device__ void getRotationMatrixCalphaDPhi(REAL *d_data, REAL phi, REAL psi, bool first){
	REAL A[16], B[16], C[16], D[16];
	if(first){
		getRotationMatrixDihedralDPsi(d_data, phi, C_N_CA, R_N_CA);
	}else{
		getRotationMatrixDihedral(B, psi, N_CA_C, R_CA_C);
		getRotationMatrixDihedral(C, OMEGACIS, CA_C_N, R_C_N);
		getRotationMatrixDihedralDPsi(A, phi, C_N_CA, R_N_CA);	
		mat44Mul(B, C, D);
		mat44Mul(D, A, d_data);	
	}
}

__device__ void getRotationMatrixCalphaDPsi(REAL *d_data, REAL phi, REAL psi){
	REAL A[16], B[16], C[16], D[16];
	getRotationMatrixDihedralDPsi(B, psi, N_CA_C, R_CA_C);
	getRotationMatrixDihedral(C, OMEGACIS, CA_C_N, R_C_N);
	getRotationMatrixDihedral(A, phi, C_N_CA, R_N_CA);
	mat44Mul(B, C, D);
	mat44Mul(D, A, d_data);	
}


__device__ void getIdentityMatrix44(REAL *d_data){
	d_data[0]=1.0;          d_data[1]=0.0;	d_data[2]=0.0;  d_data[3]=0.0;
	d_data[4]=0.0;			d_data[5]=1.0; 	d_data[6]=0.0;  d_data[7]=0.0;
	d_data[8]=0.0;   		d_data[9]=0.0; 	d_data[10]=1.0; d_data[11]=0.0;
	d_data[12]=0.0;			d_data[13]=0.0;	d_data[14]=0.0;	d_data[15]=1.0;
}

__device__ void getIdentityMatrix33(REAL *d_data){
	d_data[0]=1.0;          d_data[1]=0.0;	d_data[2]=0.0;
	d_data[3]=0.0;			d_data[4]=1.0; 	d_data[5]=0.0;
	d_data[6]=0.0;   		d_data[7]=0.0; 	d_data[8]=1.0;
}

__device__ void setMat44(REAL *d_dst, REAL d_src){
	memset(d_dst, d_src, 16*sizeof(REAL));
}
__device__ void setMat44(REAL *d_dst, REAL *d_src){
	memcpy(d_dst, d_src, 16*sizeof(REAL));
}
__device__ void setMat33(REAL *d_dst, REAL *d_src){
	memcpy(d_dst, d_src, 9*sizeof(REAL));
}

__device__ void invertMat44(REAL *d_dst, REAL *d_src){
	
	REAL trans[4], invTrans[4];
	trans[0] = d_src[3];trans[1] = d_src[7];trans[2] = d_src[11];trans[3]=1.0;

	d_dst[0]=d_src[0];	d_dst[1]=d_src[4];	d_dst[2]=d_src[8];  d_dst[3]=0;
	d_dst[4]=d_src[1];	d_dst[5]=d_src[5]; 	d_dst[6]=d_src[9];  d_dst[7]=0;
	d_dst[8]=d_src[2];  d_dst[9]=d_src[6]; 	d_dst[10]=d_src[10];d_dst[11]=0;
	d_dst[12]=0.0;		d_dst[13]=0.0;		d_dst[14]=0.0;		d_dst[15]=1.0;
	
	mat44Vec4Mul(d_dst, trans, invTrans);

	d_dst[3] = -invTrans[0]; d_dst[7] = -invTrans[1]; d_dst[11] = -invTrans[2];
}

// __device__ void mat44Mul(REAL *d_m1, REAL *d_m2, REAL *dst){
// 	if(dst == d_m1 || dst == d_m2){
// 		REAL tmp[16];
// 		for(int i=0;i<4;i++){
// 			for(int j=0;j<4;j++){
// 				tmp[i*4 + j] = 0.0;
// 				for(int k=0; k<4; k++){
// 					tmp[i*4+j] += d_m1[i*4+k]*d_m2[k*4+j];
// 				}
// 			}
// 		}
// 		memcpy(dst, tmp, 16*sizeof(REAL));
// 	}else{
// 		for(int i=0;i<4;i++){
// 			for(int j=0;j<4;j++){
// 				dst[i*4 + j] = 0.0;
// 				for(int k=0; k<4; k++){
// 					dst[i*4+j] += d_m1[i*4+k]*d_m2[k*4+j];
// 				}
// 			}
// 		}
// 	}
// }

__device__ void mat44Mul(REAL *d_m1, REAL *d_m2, REAL *dst){
	dst[0] = d_m1[0]*d_m2[0] + d_m1[1]*d_m2[4] + d_m1[2]*d_m2[8] + d_m1[3]*d_m2[12];
	dst[1] = d_m1[0]*d_m2[1] + d_m1[1]*d_m2[5] + d_m1[2]*d_m2[9] + d_m1[3]*d_m2[13];
	dst[2] = d_m1[0]*d_m2[2] + d_m1[1]*d_m2[6] + d_m1[2]*d_m2[10] + d_m1[3]*d_m2[14];
	dst[3] = d_m1[0]*d_m2[3] + d_m1[1]*d_m2[7] + d_m1[2]*d_m2[11] + d_m1[3]*d_m2[15];

	dst[4] = d_m1[4]*d_m2[0] + d_m1[5]*d_m2[4] + d_m1[6]*d_m2[8] + d_m1[7]*d_m2[12];
	dst[5] = d_m1[4]*d_m2[1] + d_m1[5]*d_m2[5] + d_m1[6]*d_m2[9] + d_m1[7]*d_m2[13];
	dst[6] = d_m1[4]*d_m2[2] + d_m1[5]*d_m2[6] + d_m1[6]*d_m2[10] + d_m1[7]*d_m2[14];
	dst[7] = d_m1[4]*d_m2[3] + d_m1[5]*d_m2[7] + d_m1[6]*d_m2[11] + d_m1[7]*d_m2[15];

	dst[8] = d_m1[8]*d_m2[0] + d_m1[9]*d_m2[4] + d_m1[10]*d_m2[8] + d_m1[11]*d_m2[12];
	dst[9] = d_m1[8]*d_m2[1] + d_m1[9]*d_m2[5] + d_m1[10]*d_m2[9] + d_m1[11]*d_m2[13];
	dst[10] = d_m1[8]*d_m2[2] + d_m1[9]*d_m2[6] + d_m1[10]*d_m2[10] + d_m1[11]*d_m2[14];
	dst[11] = d_m1[8]*d_m2[3] + d_m1[9]*d_m2[7] + d_m1[10]*d_m2[11] + d_m1[11]*d_m2[15];

	dst[12] = d_m1[12]*d_m2[0] + d_m1[13]*d_m2[4] + d_m1[14]*d_m2[8] + d_m1[15]*d_m2[12];
	dst[13] = d_m1[12]*d_m2[1] + d_m1[13]*d_m2[5] + d_m1[14]*d_m2[9] + d_m1[15]*d_m2[13];
	dst[14] = d_m1[12]*d_m2[2] + d_m1[13]*d_m2[6] + d_m1[14]*d_m2[10] + d_m1[15]*d_m2[14];
	dst[15] = d_m1[12]*d_m2[3] + d_m1[13]*d_m2[7] + d_m1[14]*d_m2[11] + d_m1[15]*d_m2[15];
}

__device__ void mat33Mul(REAL *d_m1, REAL *d_m2, REAL *dst){
	if(dst == d_m1 || dst == d_m2){
		REAL tmp[9];
		for(int i=0;i<3;i++){
			for(int j=0;j<3;j++){
				tmp[i*3 + j] = 0.0;
				for(int k=0; k<3; k++){
					tmp[i*3+j] += d_m1[i*3+k]*d_m2[k*3+j];
				}
			}
		}
		memcpy(dst, tmp, 9*sizeof(REAL));
	}else{
		for(int i=0;i<3;i++){
			for(int j=0;j<3;j++){
				dst[i*3 + j] = 0.0;
				for(int k=0; k<3; k++){
					dst[i*3+j] += d_m1[i*3+k]*d_m2[k*3+j];
				}
			}
		}
	}
}


// __device__ void mat44Vec4Mul(REAL *d_m, REAL *d_v, REAL *dst){
// 	if(dst == d_v){
// 		REAL tmp[4];
// 		for(int i=0;i<4;i++){
// 			tmp[i] = 0.0;
// 			for(int j=0;j<4;j++){
// 				tmp[i] += d_m[i*4+j]*d_v[j];
// 			}
// 		}
// 		memcpy(dst, tmp, 4*sizeof(REAL));
// 	}else{
// 		for(int i=0;i<4;i++){
// 			dst[i] = 0.0;
// 			for(int j=0;j<4;j++){
// 				dst[i] += d_m[i*4+j]*d_v[j];
// 			}
// 		}
// 	}
// }
__device__ void mat44Vec4Mul(REAL *d_m, REAL *d_v, REAL *dst){
	dst[0] = d_m[0]*d_v[0]+d_m[1]*d_v[1]+d_m[2]*d_v[2]+d_m[3]*d_v[3];
	dst[1] = d_m[4]*d_v[0]+d_m[5]*d_v[1]+d_m[6]*d_v[2]+d_m[7]*d_v[3];
	dst[2] = d_m[8]*d_v[0]+d_m[9]*d_v[1]+d_m[10]*d_v[2]+d_m[11]*d_v[3];
	dst[3] = d_m[12]*d_v[0]+d_m[13]*d_v[1]+d_m[14]*d_v[2]+d_m[15]*d_v[3];
}

// __device__ void mat33Vec3Mul(REAL *d_m, REAL *d_v, REAL *dst){
// 	if(dst == d_v){
// 		REAL tmp[3];
// 		for(int i=0;i<3;i++){
// 			tmp[i] = 0.0;
// 			for(int j=0;j<3;j++){
// 				tmp[i] += d_m[i*3+j]*d_v[j];
// 			}
// 		}
// 		memcpy(dst, tmp, 3*sizeof(REAL));
// 	}else{
// 		for(int i=0;i<3;i++){
// 			dst[i] = 0.0;
// 			for(int j=0;j<3;j++){
// 				dst[i] += d_m[i*3+j]*d_v[j];
// 			}
// 		}
// 	}
// }

__device__ void mat33Vec3Mul(REAL *d_m, REAL *d_v, REAL *dst){
	dst[0] = d_m[0]*d_v[0]+d_m[1]*d_v[1]+d_m[2]*d_v[2];
	dst[1] = d_m[3]*d_v[0]+d_m[4]*d_v[1]+d_m[5]*d_v[2];
	dst[2] = d_m[6]*d_v[0]+d_m[7]*d_v[1]+d_m[8]*d_v[2];
}

// __device__ void mat44Vec3Mul(REAL *d_m, REAL *d_v, REAL *dst){
//    REAL tmp[4], tmp1[4];
//    memcpy(tmp, d_v, 3*sizeof(REAL));tmp[3]=1.0;
//    mat44Vec4Mul(d_m, tmp, tmp1);
//    memcpy(dst, tmp1, 3*sizeof(REAL));
// }
__device__ void mat44Vec3Mul(REAL *d_m, REAL *d_v, REAL *dst){
	dst[0] = d_m[0]*d_v[0]+d_m[1]*d_v[1]+d_m[2]*d_v[2]+d_m[3];
	dst[1] = d_m[4]*d_v[0]+d_m[5]*d_v[1]+d_m[6]*d_v[2]+d_m[7];
	dst[2] = d_m[8]*d_v[0]+d_m[9]*d_v[1]+d_m[10]*d_v[2]+d_m[11];
}

__device__ void mat44Zero3Mul(REAL *d_m, REAL *dst){
	dst[0] = d_m[3];
	dst[1] = d_m[7];
	dst[2] = d_m[11];
}


__device__ void setVec3(REAL *d_v, REAL x, REAL y, REAL z){
	d_v[0]=x;d_v[1]=y;d_v[2]=z;
}

__device__ void setVec3(REAL *src, REAL *dst){
	dst[0]=src[0];dst[1]=src[1];dst[2]=src[2];
}

__device__ REAL vec3Mul(const REAL *v1, const REAL *v2){
	return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}

__device__ void extract33RotationMatrix(REAL *mat44, REAL *mat33){
	for(int i=0;i<3;i++)	
		for(int j=0;j<3;j++)
			mat33[3*i+j] = mat44[4*i+j];
}

__device__ void vec3Mul(REAL *u, REAL lambda){
	u[0]*=lambda;u[1]*=lambda;u[2]*=lambda;
}
__device__ REAL vec3Dot(REAL *v1, REAL *v2){
	return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}

__device__ void vec3Cross(REAL *u, REAL *v, REAL *w){
	w[0] = u[1]*v[2] - u[2]*v[1];
	w[1] = u[2]*v[0] - u[0]*v[2];
	w[2] = u[0]*v[1] - u[1]*v[0];
}

__device__ REAL getVec3Norm(REAL *u){
	return sqrt(vec3Dot(u,u));
}

__device__ void vec3Normalize(REAL *u){
	vec3Mul(u, 1.0/getVec3Norm(u));
}

__device__ void vec3Minus(REAL *vec1, REAL *vec2, REAL *res){
	res[0] = vec1[0]-vec2[0];res[1] = vec1[1]-vec2[1];res[2] = vec1[2]-vec2[2];
}	
__device__ void vec3Plus(REAL *vec1, REAL *vec2, REAL *res){
	res[0] = vec1[0]+vec2[0];res[1] = vec1[1]+vec2[1];res[2] = vec1[2]+vec2[2];
}	
