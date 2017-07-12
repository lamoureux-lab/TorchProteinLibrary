extern __device__ void getRotationMatrix(float *d_data, float alpha, float beta, float R);
extern __device__ void getRotationMatrixDihedral(float *d_data, float psi, float kappa, float R);
extern __device__ void getRotationMatrixDihedralDPsi(float *d_data, float psi, float kappa, float R);
// Fills rotation-translation 4x4 matrix. 
extern __device__ void getRotationMatrixDAlpha(float *d_data, float alpha, float beta, float R);
// Fills the derivative of a rotation-translation 4x4 matrix wrt alpha. 
extern __device__ void getRotationMatrixDBeta(float *d_data, float alpha, float beta, float R);
// Fills the derivative of a rotation-translation 4x4 matrix wrt beta.
extern __device__ void getIdentityMatrix44(float *d_data);
// Fills the 4x4 identity matrix.
extern __device__ void setMat44(float *d_dst, float *d_src);
// Fills the 4x4 d_dst matrix with d_src values.
extern __device__ void mat44Mul(float *d_m1, float *d_m2, float *dst);
// Multiplies two 4x4 matrixes and puts the result to dst
extern __device__ void mat44Vec4Mul(float *d_m, float *d_v, float *dst);
// Multiplies two 4x4 matrixe by 4-vector and puts the result to dst
extern __device__ void mat44Vec3Mul(float *d_m, float *d_v, float *dst);
// Multiplies two 4x4 matrixe by 3-vector and puts the result to dst
extern __device__ void setVec3(float *d_v, float x, float y, float z);
// Initializes a 3-vector
extern __device__ float vec3Mul(float *v1, float *v2);
// Multiplies two 3-vectors and returns the result



extern __device__ void get33RotationMatrix(float *d_data, float alpha, float beta);
// Fills rotation 3x3 matrix. 
extern __device__ void get33RotationMatrixDAlpha(float *d_data, float alpha, float beta);
// Fills the derivative of a rotation 3x3 matrix wrt alpha. 
extern __device__ void get33RotationMatrixDBeta(float *d_data, float alpha, float beta);
// Fills the derivative of a rotation 3x3 matrix wrt beta.
extern __device__ void getIdentityMatrix33(float *d_data);
// Fills the 3x3 identity matrix.
extern __device__ void setMat33(float *d_dst, float *d_src);
// Fills the 3x3 d_dst matrix with d_src values.
extern __device__ void mat33Mul(float *d_m1, float *d_m2, float *dst);
// Multiplies two 3x3 matrixes and puts the result to dst
extern __device__ void mat33Vec3Mul(float *d_m, float *d_v, float *dst);
// Multiplies two 3x3 matrixe by 3-vector and puts the result to dst
