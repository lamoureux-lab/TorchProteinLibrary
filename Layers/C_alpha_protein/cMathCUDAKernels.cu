#include "cMathCUDAKernels.h"
#define KAPPA1 (3.14159 - 1.9391)
#define KAPPA2 (3.14159 - 2.061)
#define KAPPA3 (3.14159 -2.1186)
#define OMEGACIS -3.1318
#define R_CALPHA_C 1.525
#define R_C_N 1.320
#define R_N_CALPHA 1.460

__device__ void getRotationMatrix(float *d_data, float alpha, float beta, float R){
	d_data[0]=cos(alpha);   d_data[1]=-sin(alpha)*cos(beta);d_data[2]=sin(alpha)*sin(beta);	d_data[3]=-R*sin(alpha)*cos(beta);
	d_data[4]=sin(alpha);	d_data[5]=cos(alpha)*cos(beta); d_data[6]=-cos(alpha)*sin(beta);d_data[7]=R*cos(alpha)*cos(beta);
	d_data[8]=0.0;   		d_data[9]=sin(beta);			d_data[10]=cos(beta); 			d_data[11]=R*sin(beta);
	d_data[12]=0.0;			d_data[13]=0.0;					d_data[14]=0.0;		 			d_data[15]=1.0;
}

__device__ void get33RotationMatrix(float *d_data, float alpha, float beta){
    d_data[0]=cos(alpha);   d_data[1]=-sin(alpha)*cos(beta);d_data[2]=sin(alpha)*sin(beta);	
	d_data[3]=sin(alpha);	d_data[4]=cos(alpha)*cos(beta); d_data[5]=-cos(alpha)*sin(beta);
	d_data[6]=0.0;   		d_data[7]=sin(beta);			d_data[8]=cos(beta); 			
}

__device__ void getRotationMatrixDAlpha(float *d_data, float alpha, float beta, float R){
	d_data[0]=-sin(alpha);  d_data[1]=-cos(alpha)*cos(beta);    d_data[2]=cos(alpha)*sin(beta);		d_data[3]=-R*cos(alpha)*cos(beta);
	d_data[4]=cos(alpha);   d_data[5]=-sin(alpha)*cos(beta); 	d_data[6]=sin(alpha)*sin(beta);		d_data[7]=-R*sin(alpha)*cos(beta);
	d_data[8]=0.0;   		d_data[9]=0.0;				 	    d_data[10]=0.0; 					d_data[11]=0.0;
	d_data[12]=0.0;			d_data[13]=0.0;					    d_data[14]=0.0;		 				d_data[15]=0.0;
}

__device__ void get33RotationMatrixDAlpha(float *d_data, float alpha, float beta){
	d_data[0]=-sin(alpha);  d_data[1]=-cos(alpha)*cos(beta);    d_data[2]=cos(alpha)*sin(beta);
	d_data[3]=cos(alpha);   d_data[4]=-sin(alpha)*cos(beta); 	d_data[5]=sin(alpha)*sin(beta);
	d_data[6]=0.0;   		d_data[7]=0.0;				 	    d_data[8]=0.0; 			
}

__device__ void getRotationMatrixDBeta(float *d_data, float alpha, float beta, float R){
	d_data[0]=0.0;          d_data[1]=sin(alpha)*sin(beta);	d_data[2]=sin(alpha)*cos(beta);		d_data[3]=R*sin(alpha)*sin(beta);
	d_data[4]=0.0;			d_data[5]=-cos(alpha)*sin(beta); 	d_data[6]=-cos(alpha)*cos(beta);    d_data[7]=-R*cos(alpha)*sin(beta);
	d_data[8]=0.0;   		d_data[9]=cos(beta); 				d_data[10]=-sin(beta); 				d_data[11]=R*cos(beta);
	d_data[12]=0.0;			d_data[13]=0.0;					    d_data[14]=0.0;		 				d_data[15]=0.0;
}

__device__ void get33RotationMatrixDBeta(float *d_data, float alpha, float beta){
	d_data[0]=0.0;          d_data[1]=sin(alpha)*sin(beta);	    d_data[2]=sin(alpha)*cos(beta);	
	d_data[3]=0.0;			d_data[4]=-cos(alpha)*sin(beta); 	d_data[5]=-cos(alpha)*cos(beta);
	d_data[6]=0.0;   		d_data[7]=cos(beta); 				d_data[8]=-sin(beta); 			
}

__device__ void getRotationMatrixDihedral(float *d_data, float psi, float kappa, float R){
	d_data[0]=cos(psi)*cos(kappa); 	d_data[1]=-cos(psi)*sin(kappa);	d_data[2]=sin(psi);	d_data[3]=0;
	d_data[4]=sin(kappa);			d_data[5]=cos(kappa); 			d_data[6]=0;		d_data[7]=R;
	d_data[8]=-sin(psi)*cos(kappa); d_data[9]=sin(psi)*sin(kappa);	d_data[10]=cos(psi);d_data[11]=0.0;
	d_data[12]=0.0;					d_data[13]=0.0;					d_data[14]=0.0;		d_data[15]=1.0;
}
__device__ void getRotationMatrixDihedralDPsi(float *d_data, float psi, float kappa, float R){
	d_data[0]=-sin(psi)*cos(kappa); 	d_data[1]=sin(psi)*sin(kappa);	d_data[2]=cos(psi);	d_data[3]=0;
	d_data[4]=0.0;						d_data[5]=0.0;	 				d_data[6]=0;		d_data[7]=0;
	d_data[8]=-cos(psi)*cos(kappa); 	d_data[9]=cos(psi)*sin(kappa);	d_data[10]=-sin(psi);d_data[11]=0.0;
	d_data[12]=0.0;						d_data[13]=0.0;					d_data[14]=0.0;		d_data[15]=0.0;
}

__device__ void getRotationMatrixCalpha(float *d_data, float phi, float psi){
	// getRotationMatrixDihedral(d_data, 0.0, psi);
	float A[16],B[16],C[16],D[16];
	getRotationMatrixDihedral(A, phi, KAPPA1, R_N_CALPHA);
	getRotationMatrixDihedral(B, psi, KAPPA2, R_CALPHA_C);
	getRotationMatrixDihedral(C, OMEGACIS, KAPPA3, R_C_N);

	mat44Mul(B, C, D);
	mat44Mul(D, A, d_data);
}

__device__ void getRotationMatrixCalphaDPhi(float *d_data, float phi, float psi){
	// getRotationMatrixDihedral(d_data, 0.0, psi);
	float A[16],B[16],C[16],D[16];
	getRotationMatrixDihedralDPsi(A, phi, KAPPA1, R_N_CALPHA);
	getRotationMatrixDihedral(B, psi, KAPPA2, R_CALPHA_C);
	getRotationMatrixDihedral(C, OMEGACIS, KAPPA3, R_C_N);

	mat44Mul(B, C, D);
	mat44Mul(D, A, d_data);
}

__device__ void getRotationMatrixCalphaDPsi(float *d_data, float phi, float psi){
	// getRotationMatrixDihedral(d_data, 0.0, psi);
	float A[16],B[16],C[16],D[16];
	getRotationMatrixDihedral(A, phi, KAPPA1, R_N_CALPHA);
	getRotationMatrixDihedralDPsi(B, psi, KAPPA2, R_CALPHA_C);
	getRotationMatrixDihedral(C, OMEGACIS, KAPPA3, R_C_N);

	mat44Mul(B, C, D);
	mat44Mul(D, A, d_data);
}


__device__ void getIdentityMatrix44(float *d_data){
	d_data[0]=1.0;          d_data[1]=0.0;	d_data[2]=0.0;  d_data[3]=0.0;
	d_data[4]=0.0;			d_data[5]=1.0; 	d_data[6]=0.0;  d_data[7]=0.0;
	d_data[8]=0.0;   		d_data[9]=0.0; 	d_data[10]=1.0; d_data[11]=0.0;
	d_data[12]=0.0;			d_data[13]=0.0;	d_data[14]=0.0;	d_data[15]=1.0;
}

__device__ void getIdentityMatrix33(float *d_data){
	d_data[0]=1.0;          d_data[1]=0.0;	d_data[2]=0.0;
	d_data[3]=0.0;			d_data[4]=1.0; 	d_data[5]=0.0;
	d_data[6]=0.0;   		d_data[7]=0.0; 	d_data[8]=1.0;
}

__device__ void setMat44(float *d_dst, float *d_src){
	memcpy(d_dst, d_src, 16*sizeof(float));
}
__device__ void setMat33(float *d_dst, float *d_src){
	memcpy(d_dst, d_src, 9*sizeof(float));
}

__device__ void mat44Mul(float *d_m1, float *d_m2, float *dst){
	if(dst == d_m1 || dst == d_m2){
		float tmp[16];
		for(int i=0;i<4;i++){
			for(int j=0;j<4;j++){
				tmp[i*4 + j] = 0.0;
				for(int k=0; k<4; k++){
					tmp[i*4+j] += d_m1[i*4+k]*d_m2[k*4+j];
				}
			}
		}
		memcpy(dst, tmp, 16*sizeof(float));
	}else{
		for(int i=0;i<4;i++){
			for(int j=0;j<4;j++){
				dst[i*4 + j] = 0.0;
				for(int k=0; k<4; k++){
					dst[i*4+j] += d_m1[i*4+k]*d_m2[k*4+j];
				}
			}
		}
	}
}

__device__ void mat33Mul(float *d_m1, float *d_m2, float *dst){
	if(dst == d_m1 || dst == d_m2){
		float tmp[9];
		for(int i=0;i<3;i++){
			for(int j=0;j<3;j++){
				tmp[i*3 + j] = 0.0;
				for(int k=0; k<3; k++){
					tmp[i*3+j] += d_m1[i*3+k]*d_m2[k*3+j];
				}
			}
		}
		memcpy(dst, tmp, 9*sizeof(float));
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


__device__ void mat44Vec4Mul(float *d_m, float *d_v, float *dst){
	if(dst == d_v){
		float tmp[4];
		for(int i=0;i<4;i++){
			tmp[i] = 0.0;
			for(int j=0;j<4;j++){
				tmp[i] += d_m[i*4+j]*d_v[j];
			}
		}
		memcpy(dst, tmp, 4*sizeof(float));
	}else{
		for(int i=0;i<4;i++){
			dst[i] = 0.0;
			for(int j=0;j<4;j++){
				dst[i] += d_m[i*4+j]*d_v[j];
			}
		}
	}
}

__device__ void mat33Vec3Mul(float *d_m, float *d_v, float *dst){
	if(dst == d_v){
		float tmp[3];
		for(int i=0;i<3;i++){
			tmp[i] = 0.0;
			for(int j=0;j<3;j++){
				tmp[i] += d_m[i*3+j]*d_v[j];
			}
		}
		memcpy(dst, tmp, 3*sizeof(float));
	}else{
		for(int i=0;i<3;i++){
			dst[i] = 0.0;
			for(int j=0;j<3;j++){
				dst[i] += d_m[i*3+j]*d_v[j];
			}
		}
	}
}

__device__ void mat44Vec3Mul(float *d_m, float *d_v, float *dst){
   float tmp[4];
   memcpy(tmp, d_v, 3*sizeof(float));tmp[3]=1.0;
   mat44Vec4Mul(d_m, tmp, dst);
}

__device__ void setVec3(float *d_v, float x, float y, float z){
	d_v[0]=x;d_v[1]=y;d_v[2]=z;
}

__device__ float vec3Mul(float *v1, float *v2){
	return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}

__device__ void extract33RotationMatrix(float *mat44, float *mat33){
	for(int i=0;i<3;i++)	
		for(int j=0;j<3;j++)
			mat33[3*i+j] = mat44[4*i+j];
}