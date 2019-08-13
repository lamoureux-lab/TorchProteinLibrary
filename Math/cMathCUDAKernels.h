#ifndef MATH_CUDA_KERNELS_H
#define MATH_CUDA_KERNELS_H

template <typename T>
__device__ void getRotationMatrix(T *d_data, T alpha, T beta, T R);
// Fills rotation-translation 4x4 matrix. 
template <typename T>
__device__ void getRotationMatrixDAlpha(T *d_data, T alpha, T beta, T R);
// Fills the derivative of a rotation-translation 4x4 matrix wrt alpha. 
template <typename T>
__device__ void getRotationMatrixDBeta(T *d_data, T alpha, T beta, T R);
// Fills the derivative of a rotation-translation 4x4 matrix wrt beta.

template <typename T>
__device__ void getRotationMatrixDihedral(T *d_data, T psi, T kappa, T R);
template <typename T>
__device__ void getRotationMatrixDihedralDPsi(T *d_data, T psi, T kappa, T R);
template <typename T>
__device__ void getRotationMatrixCalpha(T *d_data, T phi, T psi, bool first);
template <typename T>
__device__ void getRotationMatrixCalphaDPhi(T *d_data, T phi, T psi, bool first);
template <typename T>
__device__ void getRotationMatrixCalphaDPsi(T *d_data, T phi, T psi);

template <typename T>
__device__ void getIdentityMatrix44(T *d_data);
// Fills the 4x4 identity matrix.
template <typename T>
__device__ void setMat44(T *d_dst, T d_src);
template <typename T>
__device__ void setMat44(T *d_dst, T *d_src);
// Fills the 4x4 d_dst matrix with d_src values.

template <typename T>
__device__ void invertMat44(T *d_dst, T *d_src);
//Inverts 4x4 transformation matrix

template <typename T>
__device__ void mat44Mul(T *d_m1, T *d_m2, T *dst);
// Multiplies two 4x4 matrixes and puts the result to dst
template <typename T>
__device__ void mat44Vec4Mul(T *d_m, T *d_v, T *dst);
// Multiplies two 4x4 matrixe by 4-vector and puts the result to dst
template <typename T>
__device__ void mat44Vec3Mul(T *d_m, T *d_v, T *dst);
template <typename T>
__device__ void mat44Zero3Mul(T *d_m, T *dst);
// Multiplies two 4x4 matrixe by 3-vector and puts the result to dst
template <typename T>
__device__ void setVec3(T *d_v, T x, T y, T z);
// Initializes a 3-vector
template <typename T>
__device__ T vec3Mul(T *v1, T *v2);
// Multiplies two 3-vectors and returns the result
template <typename T>
__device__ void vec3Mul(T *u, T lambda);
// Multiplies 3-vector by lambda
template <typename T>
__device__ T vec3Dot(T *v1, T *v2);
// Dot product of two 3-vectors
template <typename T>
__device__ void vec3Cross(T *u, T *v, T *w);
// Cross product of two 3-vectors
template <typename T>
__device__ T getVec3Norm(T *u);
// Returns norm of a 3-vector
template <typename T>
__device__ void vec3Normalize(T *u);
// Normalizes 3-vector
template <typename T>
__device__ void vec3Minus(T *vec1, T *vec2, T *res);
// Result = Vector1 - Vector2
template <typename T>
__device__ void vec3Plus(T *vec1, T *vec2, T *res);
// Result = Vector1 + Vector2

template <typename T>
__device__ void get33RotationMatrix(T *d_data, T alpha, T beta);
// Fills rotation 3x3 matrix. 
template <typename T>
__device__ void get33RotationMatrixDAlpha(T *d_data, T alpha, T beta);
// Fills the derivative of a rotation 3x3 matrix wrt alpha. 
template <typename T>
__device__ void get33RotationMatrixDBeta(T *d_data, T alpha, T beta);
// Fills the derivative of a rotation 3x3 matrix wrt beta.
template <typename T>
__device__ void getIdentityMatrix33(T *d_data);
// Fills the 3x3 identity matrix.
template <typename T>
__device__ void setMat33(T *d_dst, T *d_src);
// Fills the 3x3 d_dst matrix with d_src values.
template <typename T>
__device__ void mat33Mul(T *d_m1, T *d_m2, T *dst);
// Multiplies two 3x3 matrixes and puts the result to dst
template <typename T>
__device__ void mat33Vec3Mul(T *d_m, T *d_v, T *dst);
// Multiplies two 3x3 matrixe by 3-vector and puts the result to dst
template <typename T>
__device__ void extract33RotationMatrix(T *mat44, T *mat33);

template <typename T>
__device__ void mat33Transpose(T *mat33, T *mat33_trans);

#endif