#ifndef MATH_CUDA_KERNELS_H
#define MATH_CUDA_KERNELS_H

extern __device__ void getRotationMatrix(double *d_data, double alpha, double beta, double R);
// Fills rotation-translation 4x4 matrix. 
extern __device__ void getRotationMatrixDAlpha(double *d_data, double alpha, double beta, double R);
// Fills the derivative of a rotation-translation 4x4 matrix wrt alpha. 
extern __device__ void getRotationMatrixDBeta(double *d_data, double alpha, double beta, double R);
// Fills the derivative of a rotation-translation 4x4 matrix wrt beta.

extern __device__ void getRotationMatrixDihedral(double *d_data, double psi, double kappa, double R);
extern __device__ void getRotationMatrixDihedralDPsi(double *d_data, double psi, double kappa, double R);
extern __device__ void getRotationMatrixCalpha(double *d_data, double phi, double psi, bool first);
extern __device__ void getRotationMatrixCalphaDPhi(double *d_data, double phi, double psi, bool first);
extern __device__ void getRotationMatrixCalphaDPsi(double *d_data, double phi, double psi);


extern __device__ void getIdentityMatrix44(double *d_data);
// Fills the 4x4 identity matrix.
extern __device__ void setMat44(double *d_dst, double d_src);
extern __device__ void setMat44(double *d_dst, double *d_src);
// Fills the 4x4 d_dst matrix with d_src values.

extern __device__ void invertMat44(double *d_dst, double *d_src);
//Inverts 4x4 transformation matrix

extern __device__ void mat44Mul(double *d_m1, double *d_m2, double *dst);
// Multiplies two 4x4 matrixes and puts the result to dst
extern __device__ void mat44Vec4Mul(double *d_m, double *d_v, double *dst);
// Multiplies two 4x4 matrixe by 4-vector and puts the result to dst
extern __device__ void mat44Vec3Mul(double *d_m, double *d_v, double *dst);
// Multiplies two 4x4 matrixe by 3-vector and puts the result to dst
extern __device__ void setVec3(double *d_v, double x, double y, double z);
// Initializes a 3-vector
extern __device__ double vec3Mul(double *v1, double *v2);
// Multiplies two 3-vectors and returns the result
__device__ void vec3Mul(double *u, double lambda);
// Multiplies 3-vector by lambda
__device__ double vec3Dot(double *v1, double *v2);
// Dot product of two 3-vectors
__device__ void vec3Cross(double *u, double *v, double *w);
// Cross product of two 3-vectors
__device__ double getVec3Norm(double *u);
// Returns norm of a 3-vector
__device__ void vec3Normalize(double *u);
// Normalizes 3-vector
__device__ void vec3Minus(double *vec1, double *vec2, double *res);
// Result = Vector1 - Vector2
__device__ void vec3Plus(double *vec1, double *vec2, double *res);
// Result = Vector1 + Vector2

extern __device__ void get33RotationMatrix(double *d_data, double alpha, double beta);
// Fills rotation 3x3 matrix. 
extern __device__ void get33RotationMatrixDAlpha(double *d_data, double alpha, double beta);
// Fills the derivative of a rotation 3x3 matrix wrt alpha. 
extern __device__ void get33RotationMatrixDBeta(double *d_data, double alpha, double beta);
// Fills the derivative of a rotation 3x3 matrix wrt beta.
extern __device__ void getIdentityMatrix33(double *d_data);
// Fills the 3x3 identity matrix.
extern __device__ void setMat33(double *d_dst, double *d_src);
// Fills the 3x3 d_dst matrix with d_src values.
extern __device__ void mat33Mul(double *d_m1, double *d_m2, double *dst);
// Multiplies two 3x3 matrixes and puts the result to dst
extern __device__ void mat33Vec3Mul(double *d_m, double *d_v, double *dst);
// Multiplies two 3x3 matrixe by 3-vector and puts the result to dst

__device__ void extract33RotationMatrix(double *mat44, double *mat33);

#endif