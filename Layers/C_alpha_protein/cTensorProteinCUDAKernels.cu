#include "cTensorProteinCUDAKernels.h"
// #include "cMathCUDAKernels.h"
#include "cMathCUDAKernels.cu"

__global__ void computeCoordinates( float *d_alpha, float *d_beta, // angles arrays
									float *d_atoms,                 //atomic coords size = atoms x 3
									float *d_A,                     //A-matrixes, saved for backward pass
									int L,                          //number of angles
									float R){                       //C-alpha dist
	setVec3(d_atoms, 0, 0, 0);
	float B[16];
	getRotationMatrix(d_A, d_alpha[0], d_beta[0], R);
	for(int i=0; i<L; i++){
		getRotationMatrix(B, d_alpha[i], d_beta[i], R);
		if(i>0){            
			mat44Mul(d_A+16*(i-1), B, d_A+16*i);
		}
		mat44Vec3Mul(d_A+16*i, d_atoms, d_atoms + 3*(i+1));
	}
}

__global__ void computeBasis( 	float *d_alpha, float *d_beta, // angles arrays
								float *d_axisx,                 //coordinates of the transformed x-axis size = atoms x 3
								float *d_axisy,                 //coordinates of the transformed y-axis size = atoms x 3
								float *d_axisz,                 //coordinates of the transformed z-axis size = atoms x 3
								float *d_B,                     //3x3 rotation matrixes, saved for backward pass
								int L){                         //number of angles
	// printf("computeBasis: start\n");
	setVec3(d_axisx, 1, 0, 0);
	setVec3(d_axisy, 0, 1, 0);
	setVec3(d_axisz, 0, 0, 1);
	float B[9];
	// printf("computeBasis: rotMat33\n");
	get33RotationMatrix(d_B, d_alpha[0], d_beta[0]);
	// printf("computeBasis: %f %f\n", d_alpha[0], d_beta[0]);
	for(int i=0; i<L; i++){
		get33RotationMatrix(B, d_alpha[i], d_beta[i]);
		if(i>0){            
			mat33Mul(d_B+9*(i-1), B, d_B+9*i);
		}
		mat33Vec3Mul(d_B+9*i, d_axisx, d_axisx + 3*(i+1));
		mat33Vec3Mul(d_B+9*i, d_axisy, d_axisy + 3*(i+1));
		mat33Vec3Mul(d_B+9*i, d_axisz, d_axisz + 3*(i+1));
		// printf("computeBasis: %d\n",i);
	}
}

__global__ void computeBasisDihedral( 	float *d_alpha, float *d_beta, // angles arrays
										float *d_axisx,                 //coordinates of the transformed x-axis size = atoms x 3
										float *d_axisy,                 //coordinates of the transformed y-axis size = atoms x 3
										float *d_axisz,                 //coordinates of the transformed z-axis size = atoms x 3
										float *d_B,                     //3x3 rotation matrixes, saved for backward pass
										int L){                         //number of angles
	setVec3(d_axisx, 1, 0, 0);
	setVec3(d_axisy, 0, 1, 0);
	setVec3(d_axisz, 0, 0, 1);
	float B[9], A[16];
	getRotationMatrixCalpha(A, d_alpha[0], d_beta[0]);
	extract33RotationMatrix(A, d_B);
	for(int i=0; i<L; i++){
		getRotationMatrixCalpha(A, d_alpha[i], d_beta[i]);
		extract33RotationMatrix(A, B);
		if(i>0){            
			mat33Mul(d_B+9*(i-1), B, d_B+9*i);
		}
		mat33Vec3Mul(d_B+9*i, d_axisx, d_axisx + 3*(i+1));
		mat33Vec3Mul(d_B+9*i, d_axisy, d_axisy + 3*(i+1));
		mat33Vec3Mul(d_B+9*i, d_axisz, d_axisz + 3*(i+1));
	}
}

__global__ void computeGradientsOptimized(
								float *d_alpha, float *d_beta, // angles arrays
								float *d_dRdAlpha, float *d_dRdBeta, //dr_j/dq_k derivatives, size=atoms x angles x 3
								float *d_A,                     //A-matrixes, computed during forward
								int L,                          //number of angles
								float R){                       //C-alpha dist
	uint k = (blockIdx.x * blockDim.x) + threadIdx.x;
	int atoms_size = L+1;
	float r_0[3];setVec3(r_0, 0, 0, 0);
	float dBdAlpha[16], dBdBeta[16], leftPartAlpha[16], leftPartBeta[16], rightPart[16];
	float tmp[16], B[16];
	getRotationMatrixDAlpha(dBdAlpha, d_alpha[k], d_beta[k], R);
	getRotationMatrixDBeta(dBdBeta, d_alpha[k], d_beta[k], R);
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
		getRotationMatrix(B, d_alpha[j], d_beta[j], R);
		mat44Mul(rightPart, B, rightPart);
	}
}

__global__ void computeBasisGradientsOptimized(
								float *d_alpha, float *d_beta, 			// angles arrays
								float *d_daxdAlpha, float *d_daxdBeta, 	//dax_j/dq_k derivatives, size=atoms x angles x 3
								float *d_daydAlpha, float *d_daydBeta, 	//day_j/dq_k derivatives, size=atoms x angles x 3
								float *d_dazdAlpha, float *d_dazdBeta, 	//daz_j/dq_k derivatives, size=atoms x angles x 3
								float *d_B,                     //3x3 rotation-matrixes, computed during forward
								int L){                          //number of angles
	int angles_size = L;
	int atoms_size = L+1;
	uint k = (blockIdx.x * blockDim.x) + threadIdx.x;

	float ax_0[3];setVec3(ax_0, 1, 0, 0);
	float ay_0[3];setVec3(ay_0, 0, 1, 0);
	float az_0[3];setVec3(az_0, 0, 0, 1);

	float dBdAlpha[9], dBdBeta[9], leftPartAlpha[9], leftPartBeta[9], rightPart[9];
	float tmp[9], B[9];
	
	get33RotationMatrixDAlpha(dBdAlpha, d_alpha[k], d_beta[k]);
	get33RotationMatrixDBeta(dBdBeta, d_alpha[k], d_beta[k]);
	if(k>0){
		mat33Mul(d_B + 9*(k-1), dBdAlpha, leftPartAlpha);
		mat33Mul(d_B + 9*(k-1), dBdBeta, leftPartBeta);
	}else{
		memcpy(leftPartAlpha, dBdAlpha, 9*sizeof(float));
		memcpy(leftPartBeta, dBdBeta, 9*sizeof(float));
	}
	getIdentityMatrix33(rightPart);
	for(int j=k+1; j<atoms_size; j++){
		int index_upper = k*atoms_size+j;
		mat33Mul(leftPartAlpha, rightPart, tmp);
		mat33Vec3Mul(tmp, ax_0, d_daxdAlpha + 3*index_upper);
		mat33Vec3Mul(tmp, ay_0, d_daydAlpha + 3*index_upper);
		mat33Vec3Mul(tmp, az_0, d_dazdAlpha + 3*index_upper);
		mat33Mul(leftPartBeta, rightPart, tmp);
		mat33Vec3Mul(tmp, ax_0, d_daxdBeta + 3*index_upper);
		mat33Vec3Mul(tmp, ay_0, d_daydBeta + 3*index_upper);
		mat33Vec3Mul(tmp, az_0, d_dazdBeta + 3*index_upper);
		get33RotationMatrix(B, d_alpha[j], d_beta[j]);
		mat33Mul(rightPart, B, rightPart);
		// printf("%f\n", d_daxdAlpha[3*index_upper]);
	}
}

__global__ void computeBasisGradientsDihedral(
								float *d_alpha, float *d_beta, 			// angles arrays
								float *d_daxdAlpha, float *d_daxdBeta, 	//dax_j/dq_k derivatives, size=atoms x angles x 3
								float *d_daydAlpha, float *d_daydBeta, 	//day_j/dq_k derivatives, size=atoms x angles x 3
								float *d_dazdAlpha, float *d_dazdBeta, 	//daz_j/dq_k derivatives, size=atoms x angles x 3
								float *d_B,                     //3x3 rotation-matrixes, computed during forward
								int L){                          //number of angles
	int angles_size = L;
	int atoms_size = L+1;
	uint k = (blockIdx.x * blockDim.x) + threadIdx.x;

	float ax_0[3];setVec3(ax_0, 1, 0, 0);
	float ay_0[3];setVec3(ay_0, 0, 1, 0);
	float az_0[3];setVec3(az_0, 0, 0, 1);

	float dBdAlpha[9], dBdBeta[9], leftPartAlpha[9], leftPartBeta[9], rightPart[9];
	float tmp[9], B[9], A[16];
	
	getRotationMatrixCalphaDPhi(A, d_alpha[k], d_beta[k]);
	extract33RotationMatrix(A, dBdAlpha);
	getRotationMatrixCalphaDPsi(A, d_alpha[k], d_beta[k]);
	extract33RotationMatrix(A, dBdBeta);
	if(k>0){
		mat33Mul(d_B + 9*(k-1), dBdAlpha, leftPartAlpha);
		mat33Mul(d_B + 9*(k-1), dBdBeta, leftPartBeta);
	}else{
		memcpy(leftPartAlpha, dBdAlpha, 9*sizeof(float));
		memcpy(leftPartBeta, dBdBeta, 9*sizeof(float));
	}
	getIdentityMatrix33(rightPart);
	for(int j=k+1; j<atoms_size; j++){
		int index_upper = k*atoms_size+j;
		mat33Mul(leftPartAlpha, rightPart, tmp);
		mat33Vec3Mul(tmp, ax_0, d_daxdAlpha + 3*index_upper);
		mat33Vec3Mul(tmp, ay_0, d_daydAlpha + 3*index_upper);
		mat33Vec3Mul(tmp, az_0, d_dazdAlpha + 3*index_upper);
		mat33Mul(leftPartBeta, rightPart, tmp);
		mat33Vec3Mul(tmp, ax_0, d_daxdBeta + 3*index_upper);
		mat33Vec3Mul(tmp, ay_0, d_daydBeta + 3*index_upper);
		mat33Vec3Mul(tmp, az_0, d_dazdBeta + 3*index_upper);
		getRotationMatrixCalpha(A, d_alpha[j], d_beta[j]);
		extract33RotationMatrix(A, B);
		mat33Mul(rightPart, B, rightPart);
		// printf("%f\n", d_daxdAlpha[3*index_upper]);
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

__global__ void backwardFromBasis(
								float *d_dalpha, float *d_dbeta, // angles gradients arrays
								float *d_dax, float *d_day, float *d_daz,      // coordinates gradients: 3 x atoms
								float *d_daxdAlpha, float *d_daxdBeta, 	//dax_j/dq_k derivatives, size=atoms x angles x 3
								float *d_daydAlpha, float *d_daydBeta, 	//day_j/dq_k derivatives, size=atoms x angles x 3
								float *d_dazdAlpha, float *d_dazdBeta, 	//daz_j/dq_k derivatives, size=atoms x angles x 3
								int L                          //number of angles
								){
	int angles_size = L;
	int atoms_size = L+1;
	uint k = (blockIdx.x * blockDim.x) + threadIdx.x;
	// d_dalpha[k]=0.0;
	// d_dbeta[k]=0.0;
	for(int j=k+1; j<atoms_size; j++){
		int index_upper = k*atoms_size+j;
		d_dalpha[k] += vec3Mul(d_dax+3*j, d_daxdAlpha + 3*index_upper);
		d_dalpha[k] += vec3Mul(d_day+3*j, d_daydAlpha + 3*index_upper);
		d_dalpha[k] += vec3Mul(d_daz+3*j, d_dazdAlpha + 3*index_upper);
		d_dbeta[k] += vec3Mul(d_dax+3*j, d_daxdBeta + 3*index_upper);
		d_dbeta[k] += vec3Mul(d_day+3*j, d_daydBeta + 3*index_upper);
		d_dbeta[k] += vec3Mul(d_daz+3*j, d_dazdBeta + 3*index_upper);
	}	
}

__global__ void computePairCoordinates(	float *coords,               // coordinates
                		                float *distances,           //pairwise coords: atoms x atoms x 3
                                		int L,                     // num angles
										int Lmax){					// max num angles
	int atoms_size = L+1;
	int atoms_max_size = Lmax + 1;
	uint k = blockIdx.x;
	uint i = threadIdx.x;
	float ri_k = coords[3*i + k];
	int plane_idx = k*atoms_max_size*atoms_max_size + i*atoms_max_size;
	for(int j=0; j<atoms_size; j++){
		distances[plane_idx + j] = ri_k - coords[3*j + k];
	}
}



__global__ void backwardFromPairCoordinates(	float *grad_coords,               // coordinates
                		                		float *grad_distances,           //pairwise coords: 3 x atoms x atoms
                                				int L,                     //num angles
												int Lmax){					// max num angles
	int atoms_size = L+1;
	int atoms_max_size = Lmax + 1;
	uint k = blockIdx.x;
	uint i = threadIdx.x;
	int plane_idx = k*atoms_max_size*atoms_max_size;
	float A_ji = 0.0;
	for(int j=0;j<atoms_size; j++){
		A_ji +=  grad_distances[plane_idx + j*atoms_max_size + i];
	}
	float A_ij = 0.0;
	for(int j=0;j<atoms_size; j++){
		A_ij +=  grad_distances[plane_idx + i*atoms_max_size + j];
	}
	grad_coords[3*i + k] = A_ij - A_ji;
}
__device__ bool isInVolume(int x_i, int y_i, int z_i, int vol_size){
	return (x_i>=0 && x_i<vol_size) && (y_i>=0 && y_i<vol_size) && (z_i>=0 && z_i<vol_size);
}
__device__ uint getFlatIndex3D(int x_i, int y_i, int z_i, int vol_size){
	return z_i + y_i*vol_size + x_i*vol_size*vol_size;
}
__global__ void gpu_computeVolumes(	int *input_types, 		//input sequence with residue types
									float *input_coords, 	//input coordinates of atoms
									float *output_volumes,  //output volumes with densities
									int L, 					//number of angles
									int num_types,			//number of residue types
									int vol_size,
									float resolution){
		int d=2;
		float shift = vol_size*resolution/2;
		uint num_atoms = L+1;
		uint a_index = blockIdx.x;
		uint a_type = threadIdx.x;
		uint vol_mem_size = vol_size*vol_size*vol_size;
		float *volume = output_volumes + a_index * num_types * vol_mem_size + a_type * vol_mem_size;
		//coordinates of the current residue
		float 	x0 = input_coords[3*a_index],
				y0 = input_coords[3*a_index + 1],
				z0 = input_coords[3*a_index + 2];
		for(int atom_idx = 0; atom_idx<num_atoms; atom_idx++){
			// printf("kernel aid : %d, %d -> %d\n",atom_idx, a_type, input_types[atom_idx]);
			if(input_types[atom_idx] != a_type) 
				continue;
			//coordinates wrt current residue
			float 	x = input_coords[3*atom_idx] - x0,
					y = input_coords[3*atom_idx + 1] - y0,
					z = input_coords[3*atom_idx + 2] - z0;
			//indexes on the grid (the vector 0,0,0 goes to the center of the grid)
			int x_i = floor(x/resolution) + vol_size/2;
			int y_i = floor(y/resolution) + vol_size/2;
			int z_i = floor(z/resolution) + vol_size/2;
			// float r2_origin = x*x+y*y+z*z; float vol_sigma = resolution*resolution*vol_size*vol_size/4;
			for(int i=x_i-d; i<=(x_i+d);i++){
				for(int j=y_i-d; j<=(y_i+d);j++){
					for(int k=z_i-d; k<=(z_i+d);k++){
						//projecting on the grid
						if( isInVolume(i, j, k, vol_size) ){
							//index of the flat memory in the volume
							int idx = getFlatIndex3D(i, j, k, vol_size);
							//coordinates of cells wrt initial atoms
							float x_loc = i*resolution - shift, y_loc = j*resolution - shift, z_loc = k*resolution - shift;
							float r2_atom =\
							(x - x_loc)*(x - x_loc)+\
							(y - y_loc)*(y - y_loc)+\
							(z - z_loc)*(z - z_loc);
							// volume[idx]+=exp(-r2_atom/2)*exp(-r2_origin/vol_sigma);
							volume[idx]+=exp(-r2_atom/2);
						}
			}}}
		}
}

__global__ void gpu_computeVolumesBackward(	int *input_types, 		//input sequence with residue types
											float *input_dvolumes, 	//input gradient of densities
											float *input_coords, 	//input coordinates
											float *output_dcoords,  //output gradients of coordinates
											int L, 					//number of angles
											int num_types,			//number of residue types
											int vol_size,			//size of the volumes
											float resolution){		//volume resolution
		
		int d=2;
		float shift = vol_size*resolution/2;
		uint num_atoms = L+1;
		uint a_index = blockIdx.x;
		uint a_type = threadIdx.x;
		uint vol_mem_size = vol_size*vol_size*vol_size;
		float *volume = input_dvolumes + a_index * num_types * vol_mem_size + a_type * vol_mem_size;
		//coordinates of the current residue
		float 	x0 = input_coords[3*a_index],
				y0 = input_coords[3*a_index + 1],
				z0 = input_coords[3*a_index + 2];
		for(int atom_idx = 0; atom_idx<num_atoms; atom_idx++){
			if(input_types[atom_idx] != a_type) 
				continue;
			//coordinates wrt current residue
			float 	x = input_coords[3*atom_idx] - x0,
					y = input_coords[3*atom_idx + 1] - y0,
					z = input_coords[3*atom_idx + 2] - z0;
			//indexes on the grid (the vector 0,0,0 goes to the center of the grid)
			int x_i = floor(x/resolution) + vol_size/2;
			int y_i = floor(y/resolution) + vol_size/2;
			int z_i = floor(z/resolution) + vol_size/2;
			// float r2_origin = x*x+y*y+z*z; float vol_sigma = resolution*resolution*vol_size*vol_size/4;
			for(int i=x_i-d; i<=(x_i+d);i++){
				for(int j=y_i-d; j<=(y_i+d);j++){
					for(int k=z_i-d; k<=(z_i+d);k++){
						if( isInVolume(i, j, k, vol_size) ){
							float L_xyz = volume[getFlatIndex3D(i, j, k, vol_size)];
							//coordinates of cells wrt initial atoms
							float x_loc = i*resolution - shift, y_loc = j*resolution - shift, z_loc = k*resolution - shift;
							float r2_atom =\
								(x - x_loc)*(x - x_loc)+\
								(y - y_loc)*(y - y_loc)+\
								(z - z_loc)*(z - z_loc);
							// float coef = L_xyz*exp(-r2_atom/2)*exp(-r2_origin/vol_sigma);
							float coef = L_xyz*exp(-r2_atom/2);
							atomicAdd(output_dcoords+3*atom_idx, -(x-x_loc)*coef);
							atomicAdd(output_dcoords+3*atom_idx+1, -(y-y_loc)*coef);
							atomicAdd(output_dcoords+3*atom_idx+2, -(z-z_loc)*coef);

							atomicAdd(output_dcoords+3*a_index, (x-x_loc)*coef);
							atomicAdd(output_dcoords+3*a_index+1, (y-y_loc)*coef);
							atomicAdd(output_dcoords+3*a_index+2, (z-z_loc)*coef);							
						}
			}}}
			// if( isInVolume(x_i, y_i, z_i, vol_size) ){
			// 	float coef = (2.0/vol_sigma)*volume[getFlatIndex3D(x_i, y_i, z_i, vol_size)]*exp(-r2_origin/vol_sigma);
			// 	atomicAdd(output_dcoords+3*atom_idx, -x*coef);
			// 	atomicAdd(output_dcoords+3*atom_idx+1, -y*coef);
			// 	atomicAdd(output_dcoords+3*atom_idx+2, -z*coef);

			// 	atomicAdd(output_dcoords+3*a_index, x*coef);
			// 	atomicAdd(output_dcoords+3*a_index+1, y*coef);
			// 	atomicAdd(output_dcoords+3*a_index+2, z*coef);
			// }
		}
}


__global__ void computeCoordinatesDihedral( 	float *d_phi, float *d_psi, // angles arrays
												float *d_atoms,                 //atomic coords size = atoms x 3
												float *d_A,                     //A-matrixes, saved for backward pass
												int L){
	setVec3(d_atoms, 0, 0, 0);
	float B[16];
	getRotationMatrixCalpha(d_A, d_phi[0], d_psi[0]);
	for(int i=0; i<L; i++){
		getRotationMatrixCalpha(B, d_phi[i], d_psi[i]);
		if(i>0){            
			mat44Mul(d_A+16*(i-1), B, d_A+16*i);
		}
		mat44Vec3Mul(d_A+16*i, d_atoms, d_atoms + 3*(i+1));
	}
}

__global__ void computeGradientsOptimizedDihedral(
								float *d_alpha, float *d_beta, // angles arrays
								float *d_dRdAlpha, float *d_dRdBeta, //dr_j/dq_k derivatives, size=atoms x angles x 3
								float *d_A,                     //A-matrixes, computed during forward
								int L){                          //number of angles
								
	uint k = (blockIdx.x * blockDim.x) + threadIdx.x;
	int atoms_size = L+1;
	float r_0[3];setVec3(r_0, 0, 0, 0);
	float dBdAlpha[16], dBdBeta[16], leftPartAlpha[16], leftPartBeta[16], rightPart[16];
	float tmp[16], B[16];
	getRotationMatrixCalphaDPhi(dBdAlpha, d_alpha[k], d_beta[k]);
	getRotationMatrixCalphaDPsi(dBdBeta, d_alpha[k], d_beta[k]);
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
		getRotationMatrixCalpha(B, d_alpha[j], d_beta[j]);
		mat44Mul(rightPart, B, rightPart);
	}
}



void cpu_computeCoordinates(float *d_alpha, float *d_beta,  // angles
							float *d_atoms,                 //atomic coords: atoms x 3
							float *d_A,                     //A-matrixes
							int L, float R){                //params
	computeCoordinates<<<1,1>>>(d_alpha, d_beta, d_atoms, d_A, L, R);
}

void cpu_computeCoordinatesDihedral(float *d_alpha, float *d_beta,  // angles
									float *d_atoms,                 //atomic coords: atoms x 3
									float *d_A,                     //A-matrixes
									int L, float R){                //params
	computeCoordinatesDihedral<<<1,1>>>(d_alpha, d_beta, d_atoms, d_A, L);
}

void cpu_computeBasis(		float *d_alpha, float *d_beta, // angles arrays
								float *d_axisx,                 //coordinates of the transformed x-axis size = atoms x 3
								float *d_axisy,                 //coordinates of the transformed y-axis size = atoms x 3
								float *d_axisz,                 //coordinates of the transformed z-axis size = atoms x 3
								float *d_B,                     //3x3 rotation matrixes, saved for backward pass
								int L){                         //number of angles
	computeBasis<<<1,1>>>(d_alpha, d_beta, d_axisx, d_axisy, d_axisz, d_B, L);
}

void cpu_computeBasisDihedral(	float *d_alpha, float *d_beta, // angles arrays
								float *d_axisx,                 //coordinates of the transformed x-axis size = atoms x 3
								float *d_axisy,                 //coordinates of the transformed y-axis size = atoms x 3
								float *d_axisz,                 //coordinates of the transformed z-axis size = atoms x 3
								float *d_B,                     //3x3 rotation matrixes, saved for backward pass
								int L){                         //number of angles
	computeBasisDihedral<<<1,1>>>(d_alpha, d_beta, d_axisx, d_axisy, d_axisz, d_B, L);
}

void cpu_computeDerivatives(float *d_alpha, float *d_beta,      // angles
							float *d_dRdAlpha, float *d_dRdBeta,//storage atoms x angles x 3
							float *d_A,                         //A-matrixes
							int L, float R){                    //params    
	computeGradientsOptimized<<<1,L>>>(d_alpha, d_beta, d_dRdAlpha, d_dRdBeta, d_A, L, R);
}

void cpu_computeDerivativesDihedral(float *d_alpha, float *d_beta,      // angles
							float *d_dRdAlpha, float *d_dRdBeta,//storage atoms x angles x 3
							float *d_A,                         //A-matrixes
							int L, float R){                    //params    
	computeGradientsOptimizedDihedral<<<1,L>>>(d_alpha, d_beta, d_dRdAlpha, d_dRdBeta, d_A, L);
}

void cpu_computeBasisGradients(			float *d_alpha, float *d_beta, 			// angles arrays
										float *d_daxdAlpha, float *d_daxdBeta, 	//dax_j/dq_k derivatives, size=atoms x angles x 3
										float *d_daydAlpha, float *d_daydBeta, 	//day_j/dq_k derivatives, size=atoms x angles x 3
										float *d_dazdAlpha, float *d_dazdBeta, 	//daz_j/dq_k derivatives, size=atoms x angles x 3
										float *d_B,                     //3x3 rotation-matrixes, computed during forward
										int L){                          //number of angles  
	computeBasisGradientsOptimized<<<1,L>>>(d_alpha, d_beta, d_daxdAlpha, d_daxdBeta, d_daydAlpha, d_daydBeta, d_dazdAlpha, d_dazdBeta, d_B, L);
}

void cpu_computeBasisGradientsDihedral(	float *d_alpha, float *d_beta, 			// angles arrays
										float *d_daxdAlpha, float *d_daxdBeta, 	//dax_j/dq_k derivatives, size=atoms x angles x 3
										float *d_daydAlpha, float *d_daydBeta, 	//day_j/dq_k derivatives, size=atoms x angles x 3
										float *d_dazdAlpha, float *d_dazdBeta, 	//daz_j/dq_k derivatives, size=atoms x angles x 3
										float *d_B,                     //3x3 rotation-matrixes, computed during forward
										int L){                          //number of angles  
	computeBasisGradientsDihedral<<<1,L>>>(d_alpha, d_beta, d_daxdAlpha, d_daxdBeta, d_daydAlpha, d_daydBeta, d_dazdAlpha, d_dazdBeta, d_B, L);
}

void cpu_backwardFromCoords(float *d_dalpha, float *d_dbeta, // angles gradients arrays
							float *d_dr,                    // coordinates gradients: 3 x atoms
							float *d_dRdAlpha, float *d_dRdBeta, //dr_j/dq_k derivatives, size=atoms x angles x 3
							int L                          //number of angles
							){                   
	backwardFromCoordinates<<<1,L>>>(d_dalpha, d_dbeta, d_dr, d_dRdAlpha, d_dRdBeta, L);
}

void cpu_backwardFromBasis(		float *d_dalpha, float *d_dbeta, // angles gradients arrays
								float *d_dax, float *d_day, float *d_daz,      // coordinates gradients: 3 x atoms
								float *d_daxdAlpha, float *d_daxdBeta, 	//dax_j/dq_k derivatives, size=atoms x angles x 3
								float *d_daydAlpha, float *d_daydBeta, 	//day_j/dq_k derivatives, size=atoms x angles x 3
								float *d_dazdAlpha, float *d_dazdBeta, 	//daz_j/dq_k derivatives, size=atoms x angles x 3
								int L                          //number of angles
							){                   
	backwardFromBasis<<<1,L>>>(d_dalpha, d_dbeta, d_dax, d_day, d_daz, d_daxdAlpha, d_daxdBeta, d_daydAlpha, d_daydBeta, d_dazdAlpha, d_dazdBeta, L);
}

void cpu_computePairCoordinates(float *coords,               // coordinates
                                float *distances,           // pairwise coords: atoms x atoms x 3
                                int L,                     // num angles
								int Lmax){					// max num angles
	
	computePairCoordinates<<<3,L+1>>>(coords, distances, L, Lmax);
}

void cpu_backwardFromPairCoordinates(	float *grad_coords,               // gradient of coordinates
                                		float *grad_distances,           // gradient of pairwise coords: atoms x atoms x 3
                               		 	int L,                     // num angles
										int Lmax){					// max num angles
	
	backwardFromPairCoordinates<<<3,L+1>>>(grad_coords, grad_distances, L, Lmax);
}

void cpu_computeVolumes(int *input_types, 		//input sequence with residue types
						float *input_coords, 	//input coordinates of atoms
						float *output_volumes,  //output volumes with densities
						int L, 					//number of angles
						int num_types,			//number of residue types
						int vol_size,			//linear size of the volumes
						float resolution){
	gpu_computeVolumes<<<L+1, num_types>>>(input_types, input_coords, output_volumes, L, num_types, vol_size, resolution);
}

void cpu_computeVolumesBackward(	int *input_types, 		//input sequence with residue types
									float *input_dvolumes, 	//input gradient of densities
									float *input_coords, 	//input coordinates
									float *output_dcoords,  //output gradients of coordinates
									int L, 					//number of angles
									int num_types,			//number of residue types
									int vol_size,			//size of the volumes
									float resolution){		//volume resolution
	gpu_computeVolumesBackward<<<L+1,num_types>>>(input_types, input_dvolumes, input_coords, output_dcoords, L, num_types, vol_size, resolution);
}