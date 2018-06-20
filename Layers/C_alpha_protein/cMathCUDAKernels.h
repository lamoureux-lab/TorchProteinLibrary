#ifndef MATH_CUDA_KERNELS_H
#define MATH_CUDA_KERNELS_H

#define REAL float

extern __device__ void getRotationMatrix(REAL *d_data, REAL alpha, REAL beta, REAL R);
// Fills rotation-translation 4x4 matrix. 
extern __device__ void getRotationMatrixDAlpha(REAL *d_data, REAL alpha, REAL beta, REAL R);
// Fills the derivative of a rotation-translation 4x4 matrix wrt alpha. 
extern __device__ void getRotationMatrixDBeta(REAL *d_data, REAL alpha, REAL beta, REAL R);
// Fills the derivative of a rotation-translation 4x4 matrix wrt beta.

extern __device__ void getRotationMatrixDihedral(REAL *d_data, REAL psi, REAL kappa, REAL R);
extern __device__ void getRotationMatrixDihedralDPsi(REAL *d_data, REAL psi, REAL kappa, REAL R);
extern __device__ void getRotationMatrixCalpha(REAL *d_data, REAL phi, REAL psi, bool first);
extern __device__ void getRotationMatrixCalphaDPhi(REAL *d_data, REAL phi, REAL psi, bool first);
extern __device__ void getRotationMatrixCalphaDPsi(REAL *d_data, REAL phi, REAL psi);


extern __device__ void getIdentityMatrix44(REAL *d_data);
// Fills the 4x4 identity matrix.
extern __device__ void setMat44(REAL *d_dst, REAL d_src);
extern __device__ void setMat44(REAL *d_dst, REAL *d_src);
// Fills the 4x4 d_dst matrix with d_src values.

extern __device__ void invertMat44(REAL *d_dst, REAL *d_src);
//Inverts 4x4 transformation matrix

extern __device__ void mat44Mul(REAL *d_m1, REAL *d_m2, REAL *dst);
// Multiplies two 4x4 matrixes and puts the result to dst
extern __device__ void mat44Vec4Mul(REAL *d_m, REAL *d_v, REAL *dst);
// Multiplies two 4x4 matrixe by 4-vector and puts the result to dst
extern __device__ void mat44Vec3Mul(REAL *d_m, REAL *d_v, REAL *dst);
// Multiplies two 4x4 matrixe by 3-vector and puts the result to dst
extern __device__ void setVec3(REAL *d_v, REAL x, REAL y, REAL z);
// Initializes a 3-vector
extern __device__ REAL vec3Mul(REAL *v1, REAL *v2);
// Multiplies two 3-vectors and returns the result
__device__ void vec3Mul(REAL *u, REAL lambda);
// Multiplies 3-vector by lambda
__device__ REAL vec3Dot(REAL *v1, REAL *v2);
// Dot product of two 3-vectors
__device__ void vec3Cross(REAL *u, REAL *v, REAL *w);
// Cross product of two 3-vectors
__device__ REAL getVec3Norm(REAL *u);
// Returns norm of a 3-vector
__device__ void vec3Normalize(REAL *u);
// Normalizes 3-vector
__device__ void vec3Minus(REAL *vec1, REAL *vec2, REAL *res);
// Result = Vector1 - Vector2
__device__ void vec3Plus(REAL *vec1, REAL *vec2, REAL *res);
// Result = Vector1 + Vector2

extern __device__ void get33RotationMatrix(REAL *d_data, REAL alpha, REAL beta);
// Fills rotation 3x3 matrix. 
extern __device__ void get33RotationMatrixDAlpha(REAL *d_data, REAL alpha, REAL beta);
// Fills the derivative of a rotation 3x3 matrix wrt alpha. 
extern __device__ void get33RotationMatrixDBeta(REAL *d_data, REAL alpha, REAL beta);
// Fills the derivative of a rotation 3x3 matrix wrt beta.
extern __device__ void getIdentityMatrix33(REAL *d_data);
// Fills the 3x3 identity matrix.
extern __device__ void setMat33(REAL *d_dst, REAL *d_src);
// Fills the 3x3 d_dst matrix with d_src values.
extern __device__ void mat33Mul(REAL *d_m1, REAL *d_m2, REAL *dst);
// Multiplies two 3x3 matrixes and puts the result to dst
extern __device__ void mat33Vec3Mul(REAL *d_m, REAL *d_v, REAL *dst);
// Multiplies two 3x3 matrixe by 3-vector and puts the result to dst

__device__ void extract33RotationMatrix(REAL *mat44, REAL *mat33);

#endif