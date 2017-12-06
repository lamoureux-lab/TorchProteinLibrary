#include "cTensorProteinCUDAKernels.h"
// #include "cMathCUDAKernels.h"
#include "cMathCUDAKernels.cu"


#define EPS 1E-3

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
	float u[3], v[3], w[3], a[3], b[3], r[3], lambda;

	vec3Minus(d_coords + 3*(angle_id+1), d_coords + 3*(angle_id), v);
	if(angle_id>0)
		vec3Minus(d_coords + 3*(angle_id-1), d_coords + 3*(angle_id), u);
	else
		setVec3(u, -1, 0, 0);
	vec3Normalize(u);
	vec3Normalize(v);

	// if( atom_id<(angle_id-1) || atom_id>(angle_id+1) )
	// 	return;
	if(atom_id<(angle_id)){
		return;
		vec3Minus(d_coords + 3*(atom_id), d_coords + 3*(angle_id), r);
		compute_w(v,u,w);// vec3Cross(v, u, w);
		vec3Cross(w, r, a);
		lambda = vec3Dot(r,r) - vec3Dot(r,w);
		if(lambda>EPS)
			vec3Mul(a,-1./lambda);
		else
			vec3Mul(a,0.);
		setVec3(a, d_B_bend+deriv_flat_index);
	}else if(atom_id>(angle_id)){
		// return;
		vec3Minus(d_coords + 3*(atom_id), d_coords + 3*(angle_id), r);
		compute_w(u,v,w); //vec3Cross(u, v, w);
		vec3Cross(w, r, a);
		lambda = vec3Dot(r,r) - vec3Dot(r,w);
		if(fabs(lambda)>EPS)
			vec3Mul(a,-1./lambda);
		else
			return;
		
		setVec3(a, d_B_bend+deriv_flat_index);
	}else{
		return;
		if(angle_id>0)
			vec3Minus(d_coords + 3*(angle_id), d_coords + 3*(angle_id-1), r);
		else
			setVec3(r, 1,0,0);
		compute_w(u,v,w); //vec3Cross(u, v, w);
		vec3Cross(w, r, a);
		lambda = vec3Dot(r,r) - vec3Dot(r,w);
		if(lambda>EPS)
			vec3Mul(a,1./lambda);
		else
			vec3Mul(a,0.);
		vec3Minus(d_coords + 3*(angle_id), d_coords + 3*(angle_id+1), r);
		compute_w(v,u,w); //vec3Cross(v, u, w);
		vec3Cross(w, r, b);
		lambda = vec3Dot(r,r) - vec3Dot(r,w);
		if(lambda>EPS)
			vec3Mul(b,-1./lambda);
		else
			vec3Mul(b,0.);
		// vec3Plus(a, b, d_B_bend+deriv_flat_index);
		setVec3(a,d_B_bend+deriv_flat_index);
		// printf("a = %f, %f, %f\n", a[0], a[1], a[2]);
	}
}

__global__ void computeBMatrixRot( float *d_alpha, float *d_beta,
									float *d_coords,
									float *d_B_rot,
									int L){
	int angles_size = L;
	int atoms_size = L+1;
	uint angle_id = blockIdx.x;
	uint atom_id = threadIdx.x;
	uint deriv_flat_index = angle_id *(atoms_size*3) + atom_id*3;
	float u[3], v[3], w[3], a[3], b[3], r[3], lambda;

	
	if(atom_id<(angle_id)){
		return;
		vec3Minus(d_coords + 3*(atom_id), d_coords + 3*(angle_id+1), r);
		vec3Minus(d_coords + 3*(angle_id), d_coords + 3*(angle_id+1), w);
		vec3Normalize(w);
		vec3Cross(w, r, a);
		lambda = vec3Dot(r,r) - vec3Dot(r,w);
		if(lambda>EPS)
			vec3Mul(a,1./lambda);
		else
			vec3Mul(a,0.);
		setVec3(a, d_B_rot+deriv_flat_index);
	}else if(atom_id>(angle_id+1)){
		vec3Minus(d_coords + 3*(atom_id), d_coords + 3*(angle_id+1), r);
		vec3Minus(d_coords + 3*(angle_id+1), d_coords + 3*(angle_id), w);
		vec3Normalize(w);
		vec3Cross(w, r, a);
		lambda = vec3Dot(r,r) - vec3Dot(r,w);
		if(fabs(lambda)>EPS)
			vec3Mul(a,1./lambda);
		else
			return;
		
		setVec3(a, d_B_rot+deriv_flat_index);
	}else{
		return;
		float uw[3], vw[3], wv[3], mod_uw, mod_wv, beta;
		beta = d_beta[angle_id];
		if (fabs(sin(beta))<EPS){
			setVec3(d_B_rot+deriv_flat_index, 0.0, 0.0, 0.0);
			return;
		}else{
			if(angle_id==0)
				setVec3(u, 0, -1, 0);
			else
				vec3Minus(d_coords + 3*(angle_id-1), d_coords + 3*(angle_id), u);
			vec3Minus(d_coords + 3*(angle_id), d_coords + 3*(angle_id+1), w);
			vec3Minus(d_coords + 3*(angle_id+2), d_coords + 3*(angle_id+1), v);
			//computing a
			vec3Cross(u, w, uw);
			vec3Cross(w, v, wv);
			mod_uw = getVec3Norm(uw);
			mod_wv = getVec3Norm(wv);
			vec3Mul(wv, -1.0/(mod_uw*mod_wv));
			vec3Mul(uw, -cos(beta)/(mod_uw*mod_uw));
			vec3Plus(wv, uw, a);
			//computing b
			vec3Cross(u, w, uw);
			vec3Cross(w, v, wv);
			mod_uw = getVec3Norm(uw);
			mod_wv = getVec3Norm(wv);
			vec3Mul(uw, -1.0/(mod_uw*mod_wv));
			vec3Mul(uw, +cos(beta)/(mod_wv*mod_wv));
			vec3Plus(wv, uw, b);
			//computing derivative
			if(atom_id == angle_id){
				float u_p_w[3], u_p_w_x_a[3], vb[3];
				vec3Plus(u, w, u_p_w);
				vec3Cross(u_p_w, a, u_p_w_x_a);
				vec3Cross(v,b, vb);
				vec3Plus(u_p_w_x_a, vb, d_B_rot+deriv_flat_index);
				vec3Mul(d_B_rot+deriv_flat_index, 1.0/sin(beta));
			}
			if(atom_id == (angle_id+1)){
				float v_m_w[3], v_m_w_x_b[3], ua[3];
				vec3Minus(v, w, v_m_w);
				vec3Cross(v_m_w, b, v_m_w_x_b);
				vec3Cross(u,a, ua);
				vec3Plus(v_m_w_x_b, ua, d_B_rot+deriv_flat_index);
				vec3Mul(d_B_rot+deriv_flat_index, -1.0/sin(beta));
			}
		}
	}
}

__global__ void computeBMatrixBend_MODELLER( 	float *d_alpha, float *d_beta,
												float *d_coords,
												float *d_B_bend,
												int L){
	int angles_size = L;
	int atoms_size = L+1;
	uint angle_id = threadIdx.x;
	uint atom_id = angle_id;
	uint deriv_i_flat_index = angle_id *(atoms_size*3) + (atom_id-1)*3;
	uint deriv_j_flat_index = angle_id *(atoms_size*3) + (atom_id)*3;
	uint deriv_k_flat_index = angle_id *(atoms_size*3) + (atom_id+1)*3;
	float rij[3], rkj[3], rkj_cosa[3], rij_cosa[3], mod_rij, mod_rkj;
	float alpha = d_alpha[angle_id];

	if(angle_id>0)
		vec3Minus(d_coords + 3*(atom_id-1), d_coords + 3*(atom_id), rij);
	else
		setVec3(rij, -1,0,0);
	vec3Minus(d_coords + 3*(atom_id+1), d_coords + 3*(atom_id), rkj);
	mod_rij = getVec3Norm(rij);
	mod_rkj = getVec3Norm(rkj);
	vec3Normalize(rij);
	vec3Normalize(rkj);
	setVec3(rij, rij_cosa);
	vec3Mul(rij_cosa, -cos(alpha));
	setVec3(rkj, rkj_cosa);
	vec3Mul(rkj_cosa, -cos(alpha));

	float coef1 = sqrt(1.0 - cos(alpha)*cos(alpha))*mod_rij;
	if(coef1<EPS)
		coef1 = 0.0;
	else 
		coef1 = -cos(alpha)/coef1;
	if(angle_id>0){
		vec3Minus(rij_cosa, rkj, d_B_bend + deriv_i_flat_index);
		vec3Mul(d_B_bend + deriv_i_flat_index, coef1);
	}

	float coef2 = sqrt(1.0 - cos(alpha)*cos(alpha))*mod_rkj;
	if(coef2<EPS)
		coef2 = 0.0;
	else 
		coef2 = -cos(alpha)/coef2;
	
	vec3Minus(rkj_cosa, rij, d_B_bend + deriv_k_flat_index);
	vec3Mul(d_B_bend + deriv_k_flat_index, coef2);
	
	if(angle_id>0){
		vec3Plus(d_B_bend + deriv_k_flat_index, d_B_bend + deriv_i_flat_index, d_B_bend + deriv_j_flat_index);
		vec3Mul(d_B_bend + deriv_j_flat_index, -1.0);
	}else{
		setVec3(d_B_bend + deriv_k_flat_index, d_B_bend + deriv_j_flat_index);
		vec3Mul(d_B_bend + deriv_j_flat_index, -1.0);
	}
}


__global__ void computeBMatrixRot_MODELLER( 	float *d_alpha, float *d_beta,
												float *d_coords,
												float *d_B_rot,
												int L){
	int angles_size = L;
	int atoms_size = L+1;
	uint angle_id = threadIdx.x;
	uint atom_id = angle_id; // j
	uint deriv_i_flat_index = angle_id *(atoms_size*3) + (atom_id-1)*3;
	uint deriv_j_flat_index = angle_id *(atoms_size*3) + (atom_id)*3;
	uint deriv_k_flat_index = angle_id *(atoms_size*3) + (atom_id+1)*3;
	uint deriv_l_flat_index = angle_id *(atoms_size*3) + (atom_id+2)*3;
	float rij[3], rkl[3], rkj[3], rik[3], rjl[3];

	vec3Minus(d_coords + 3*(atom_id-1), d_coords + 3*(atom_id), rij);
	vec3Minus(d_coords + 3*(atom_id+1), d_coords + 3*(atom_id), rkj);
	vec3Minus(d_coords + 3*(atom_id+2), d_coords + 3*(atom_id+1), rkl);
	vec3Minus(d_coords + 3*(atom_id-1), d_coords + 3*(atom_id+1), rik);
	vec3Minus(d_coords + 3*(atom_id), d_coords + 3*(atom_id+2), rjl);

	float beta;
	beta = d_beta[angle_id];
	if (fabs(sin(beta))<EPS){
		return;
	}else{
		float b[3], a[3], r_kj_kl[3], r_ij_kj[3], mod_ij_kj, mod_kj_kl;
		vec3Cross(rkj, rkl, r_kj_kl);
		vec3Cross(rij, rkj, r_ij_kj);
		mod_ij_kj = getVec3Norm(r_ij_kj);
		vec3Normalize(r_kj_kl);
		vec3Normalize(r_ij_kj);
		vec3Mul(r_ij_kj, -cos(beta));
		vec3Plus(r_kj_kl, r_ij_kj, a);
		vec3Mul(a, 1.0/mod_ij_kj);

		vec3Cross(rij, rkj, r_ij_kj);
		vec3Cross(rkj, rkl, r_kj_kl);
		mod_kj_kl = getVec3Norm(r_kj_kl);
		vec3Normalize(r_kj_kl);
		vec3Normalize(r_ij_kj);
		vec3Mul(r_kj_kl, -cos(beta));
		vec3Plus(r_ij_kj, r_kj_kl, b);
		vec3Mul(b, 1.0/mod_kj_kl);

		float dcos_dri[3];
		vec3Cross(rkj, a, dcos_dri);
		vec3Mul(dcos_dri, -1.0/sin(beta));
		setVec3(dcos_dri, d_B_rot + deriv_i_flat_index);

		float dcos_drj[3], r_ika[3], r_klb[3];
		vec3Cross(rik, a, r_ika);
		vec3Cross(rkl, b, r_klb);
		vec3Mul(r_klb, -1.);
		vec3Plus(r_ika, r_klb, dcos_drj);
		vec3Mul(dcos_drj, -1.0/sin(beta));
		setVec3(dcos_drj, d_B_rot + deriv_j_flat_index);

		float dcos_drk[3], r_jlb[3], r_ija[3];
		vec3Cross(rjl, b, r_jlb);
		vec3Cross(rij, a, r_ija);
		vec3Mul(r_ija, -1.);
		vec3Plus(r_jlb, r_ija, dcos_drk);
		vec3Mul(dcos_drk, -1.0/sin(beta));
		setVec3(dcos_drk, d_B_rot + deriv_k_flat_index);

		float dcos_drl[3];
		vec3Cross(rij, b, dcos_drl);
		vec3Mul(dcos_drl, -1.0/sin(beta));
		setVec3(dcos_drl, d_B_rot + deriv_l_flat_index);
	}
}


__global__ void computeDAngles( float *d_forces,
								float *d_dangles,
								float *d_B_bend,	// L x L + 1 x 3 matrix
								float *d_B_rot,	// L x L + 1 x 3 matrix
								int L){
	int angles_size = L;
	int atoms_size = L+1;
	uint angle_type = blockIdx.x;
	uint angle_id = threadIdx.x;


	uint deriv_flat_index_all = angle_id *(atoms_size*3);
	uint deriv_flat_index;
	for(uint atom_id=0; atom_id<atoms_size; atom_id++){
		deriv_flat_index = deriv_flat_index_all + atom_id*3;
		if(angle_type==0)
			d_dangles[angle_type*angles_size + angle_id] += vec3Dot(d_forces+atom_id*3, d_B_bend+deriv_flat_index);
		else
			d_dangles[angle_type*angles_size + angle_id] += vec3Dot(d_forces+atom_id*3, d_B_rot+deriv_flat_index);
	}
}


__global__ void computeForces( 	float *d_ddist, // atoms_size x atoms_size
								float *d_forces, // atoms_size x 3
								float *d_coords, // atoms_size x 3
								int L){
	int angles_size = L;
	int atoms_size = L+1;
	uint atom_id1 = threadIdx.x;

	uint ddist_flat_index;
	float rij[3];
	for(uint atom_id2=0; atom_id2<atoms_size; atom_id2++){
		if(atom_id1<=atom_id2) continue;
		ddist_flat_index = atom_id1*atoms_size + atom_id2;
		vec3Minus(d_coords+atom_id1*3, d_coords+atom_id2*3, rij);
		vec3Normalize(rij);
		vec3Mul(rij, d_ddist[ddist_flat_index]);
		vec3Plus(d_forces+atom_id1*3, rij, d_forces+atom_id1*3);
	}
}


void cpu_computeBMatrix( 	float *d_alpha, float *d_beta,
							float *d_coords,
							float *d_B_bend,	// L x L + 1 x 3 matrix
							float *d_B_rot,	// L x L + 1 x 3 matrix
							int L){
	computeBMatrixBend<<<L,L+1>>>(d_alpha, d_beta, d_coords, d_B_bend, L);
	computeBMatrixRot<<<L,L+1>>>(d_alpha, d_beta, d_coords, d_B_rot, L);
	// computeBMatrixBend_MODELLER<<<1, L>>>(d_alpha, d_beta, d_coords, d_B_bend, L);
	// computeBMatrixBend_MODELLER<<<1, L-1>>>(d_alpha, d_beta, d_coords, d_B_bend, L);
}

void cpu_computeDAngles( 	float *d_forces,
							float *d_dangles,
							float *d_B_bend,	// L x L + 1 x 3 matrix
							float *d_B_rot,	// L x L + 1 x 3 matrix
							int L){
	computeDAngles<<<2,L>>>(d_forces, d_dangles, d_B_bend, d_B_rot, L);
}

void cpu_computeForces( 	float *d_ddist,
							float *d_forces,
							float *d_coords,
							int L){
	computeForces<<<1,L+1>>>(d_ddist, d_forces, d_coords, L);
}