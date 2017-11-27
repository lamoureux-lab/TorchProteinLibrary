#include "cTensorProteinCUDAKernels.h"
// #include "cMathCUDAKernels.h"
#include "cMathCUDAKernels.cu"


#define EPS 1E-5

__device__ void getABRotationMatrix(float *d_data, float alpha, float beta){
	d_data[0]=cos(alpha);   d_data[1]=-sin(alpha)*cos(beta);d_data[2]=sin(alpha)*sin(beta);	d_data[3]=cos(alpha);
	d_data[4]=sin(alpha);	d_data[5]=cos(alpha)*cos(beta); d_data[6]=-cos(alpha)*sin(beta);d_data[7]=sin(alpha);
	d_data[8]=0.0;   		d_data[9]=sin(beta);			d_data[10]=cos(beta); 			d_data[11]=0.0;
	d_data[12]=0.0;			d_data[13]=0.0;					d_data[14]=0.0;		 			d_data[15]=1.0;
}

__device__ void getABRotationMatrixDAlpha(float *d_data, float alpha, float beta){
	d_data[0]=-sin(alpha);  d_data[1]=-cos(alpha)*cos(beta);    d_data[2]=cos(alpha)*sin(beta);		d_data[3]=-sin(alpha);
	d_data[4]=cos(alpha);   d_data[5]=-sin(alpha)*cos(beta); 	d_data[6]=sin(alpha)*sin(beta);		d_data[7]=cos(alpha);
	d_data[8]=0.0;   		d_data[9]=0.0;				 	    d_data[10]=0.0; 					d_data[11]=0.0;
	d_data[12]=0.0;			d_data[13]=0.0;					    d_data[14]=0.0;		 				d_data[15]=0.0;
}

__device__ void getABRotationMatrixDBeta(float *d_data, float alpha, float beta){
	d_data[0]=0.0;          d_data[1]=sin(alpha)*sin(beta);	d_data[2]=sin(alpha)*cos(beta);		d_data[3]=0.0;
	d_data[4]=0.0;			d_data[5]=-cos(alpha)*sin(beta); 	d_data[6]=-cos(alpha)*cos(beta);    d_data[7]=0.0;
	d_data[8]=0.0;   		d_data[9]=cos(beta); 				d_data[10]=-sin(beta); 				d_data[11]=0.0;
	d_data[12]=0.0;			d_data[13]=0.0;					    d_data[14]=0.0;		 				d_data[15]=0.0;
}


__global__ void computeCoordinates( float *d_alpha, float *d_beta, // angles arrays
									float *d_atoms,                 //atomic coords size = atoms x 3
									float *d_A,                     //A-matrixes, saved for backward pass
									int L                          //number of angles
									){
	setVec3(d_atoms, 0, 0, 0);
	float B[16];
	getABRotationMatrix(d_A, d_alpha[0], d_beta[0]);
	for(int i=0; i<L; i++){
		getABRotationMatrix(B, d_alpha[i], d_beta[i]);
		if(i>0){            
			mat44Mul(d_A+16*(i-1), B, d_A+16*i);
		}
		mat44Vec3Mul(d_A+16*i, d_atoms, d_atoms + 3*(i+1));
	}
}

__global__ void computeGradientsOptimized(
								float *d_alpha, float *d_beta, // angles arrays
								float *d_dRdAlpha, float *d_dRdBeta, //dr_j/dq_k derivatives, size=atoms x angles x 3
								float *d_A,                     //A-matrixes, computed during forward
								int L                          //number of angles
								){
	uint k = (blockIdx.x * blockDim.x) + threadIdx.x;
	int atoms_size = L+1;
	float r_0[3];setVec3(r_0, 0, 0, 0);
	float dBdAlpha[16], dBdBeta[16], leftPartAlpha[16], leftPartBeta[16], rightPart[16];
	float tmp[16], B[16];
	getABRotationMatrixDAlpha(dBdAlpha, d_alpha[k], d_beta[k]);
	getABRotationMatrixDBeta(dBdBeta, d_alpha[k], d_beta[k]);
	if(k>0){
		mat44Mul(d_A + 16*(k-1), dBdAlpha, leftPartAlpha);
		mat44Mul(d_A + 16*(k-1), dBdBeta, leftPartBeta);
	}else{
		memcpy(leftPartAlpha, dBdAlpha, 16*sizeof(float));
		memcpy(leftPartBeta, dBdBeta, 16*sizeof(float));
	}
	getIdentityMatrix44(rightPart);
	for(int j=k+1; j<atoms_size; j++){
		int index_upper = k*atoms_size+j;
		mat44Mul(leftPartAlpha, rightPart, tmp);
		mat44Vec3Mul(tmp, r_0, d_dRdAlpha + 3*index_upper);
		mat44Mul(leftPartBeta, rightPart, tmp);
		mat44Vec3Mul(tmp, r_0, d_dRdBeta + 3*index_upper);
		getABRotationMatrix(B, d_alpha[j], d_beta[j]);
		mat44Mul(rightPart, B, rightPart);
	}
}

__global__ void backwardFromCoordinates(
								float *d_dalpha, float *d_dbeta, // angles gradients arrays
								float *d_dr,                    // coordinates gradients: 3 x atoms
								float *dRdAlpha, float *dRdBeta, //dr_j/dq_k derivatives, size=atoms x angles x 3
								int L                          //number of angles
								){
	int angles_size = L;
	int atoms_size = L+1;
	uint k = (blockIdx.x * blockDim.x) + threadIdx.x;
	// d_dalpha[k]=0.0;
	// d_dbeta[k]=0.0;
	for(int j=k+1; j<atoms_size; j++){
		int index_upper = k*atoms_size+j;
		d_dalpha[k] += vec3Mul(d_dr+3*j, dRdAlpha + 3*index_upper);
		d_dbeta[k] += vec3Mul(d_dr+3*j, dRdBeta + 3*index_upper);
	}
}

void cpu_computeCoordinates(float *d_alpha, float *d_beta,  // angles
							float *d_atoms,                 //atomic coords: atoms x 3
							float *d_A,                     //A-matrixes
							int L){                //params
	computeCoordinates<<<1,1>>>(d_alpha, d_beta, d_atoms, d_A, L);
}

void cpu_computeDerivatives(float *d_alpha, float *d_beta,      // angles
							float *d_dRdAlpha, float *d_dRdBeta,//storage atoms x angles x 3
							float *d_A,                         //A-matrixes
							int L){                    //params    
	computeGradientsOptimized<<<1,L>>>(d_alpha, d_beta, d_dRdAlpha, d_dRdBeta, d_A, L);
}

void cpu_backwardFromCoords(float *d_dalpha, float *d_dbeta, // angles gradients arrays
							float *d_dr,                    // coordinates gradients: 3 x atoms
							float *d_dRdAlpha, float *d_dRdBeta, //dr_j/dq_k derivatives, size=atoms x angles x 3
							int L                          //number of angles
							){                   
	backwardFromCoordinates<<<1,L>>>(d_dalpha, d_dbeta, d_dr, d_dRdAlpha, d_dRdBeta, L);
}


__device__ void compute_w(float *u, float *v, float *w){
	float product = u[0]*v[0]+u[1]*v[1]+u[2]*v[2];
	if(abs(product)>EPS){
		vec3Cross(u,v,w);
	}else{
		float v1[3]={1,-1,1}; //chosing an arbitrary vector (1, -1, 1)
		if( (vec3Dot(u,v1)>EPS) && (vec3Dot(v,v1)>EPS) ){
			vec3Cross(u,v1,w);
		}else{ //another arbitrary vector (-1, 1, 1)
			float v2[3]={-1,1,1};
			vec3Cross(u,v2,w);
		}
	}
	vec3Normalize(w);
}


__global__ void computeBMatrixBend( float *d_alpha, float *d_beta,
									float *d_coords,
									float *d_B_bend,
									int L){
	int angles_size = L;
	int atoms_size = L+1;
	uint angle_id = blockIdx.x;
	uint atom_id = threadIdx.x;
	uint deriv_flat_index = angle_id *(atoms_size*3) + atom_id*3;
	float u[3], v[3], w[3];
	float lambda_u, lambda_v;
	uint vector_id_center, vector_id_right, vector_id_left;
	float a[3], b[3];

	uint m = angle_id;
	uint o = angle_id+1;
	uint n = angle_id+2;
	float xi1=0, xi2=0;
	if(atom_id<=m){
		vector_id_left = atom_id;
		vector_id_center = o;
		vector_id_right = n;
		xi1 = 1;
	}else if(atom_id>=n)
		vector_id_left = m;
		vector_id_center = o;
		vector_id_right = atom_id;
		xi2 = 1;
	}else{
		vector_id_left = m;
		vector_id_center = atom_id;
		vector_id_right = n;
		xi1 = -1;
		xi2 = -1;
	}
	vec3Minus(d_coords + 3*vector_id_left, d_coords + 3*vector_id_center, u);
	vec3Minus(d_coords + 3*vector_id_right, d_coords + 3*vector_id_center, v);
	lambda_u = getVec3Norm(u);
	lambda_v = getVec3Norm(v);
	vec3Normalize(u);
	vec3Normalize(v);
	
	compute_w(u, v, w);

	vec3Cross(u, w, a);
	vec3Mul(a, xi1/lambda_u);
	vec3Cross(w, v, b);
	vec3Mul(b, xi2/lambda_v);
	vec3Plus(a, b, d_B_bend+deriv_flat_index);
}


__global__ void computeBMatrixRot(  float *d_alpha, float *d_beta,
									float *d_coords,
									float *d_B_rot,
									int L){

	int angles_size = L;
	int atoms_size = L+1;
	uint angle_id = blockIdx.x;
	uint atom_id = threadIdx.x;
	uint deriv_flat_index = angle_id *(atoms_size*3) + atom_id*3;


}


void cpu_computeBMatrix( 	float *d_dalpha, float *d_dbeta,
							float *d_coords,
							float *d_B_bend,	// L x L + 1 x 3 matrix
							float *d_B_rot,	// L x L + 1 x 3 matrix
							int L){
	computeBMatrixBend<<<L,L+1>>>(d_alpha, d_beta, d_B_bend, L);
	computeBMatrixRot<<<L,L+1>>>(d_alpha, d_beta, d_B_rot, L);
}