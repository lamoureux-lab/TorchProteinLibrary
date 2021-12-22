import math as m
import numpy as np

# functions from cVector3.cpp
#function to create a 3d Vector


def vector3(x=0, y=0, z=0):
    v = np.array([[x, y, z]])
    return v


# functions from cMatrix33.cpp
#function to create a 3x3 Matrix
def matrix33(m00=0, m01=0, m02=0,
             m10=0, m11=0, m12=0,
             m20=0, m21=0, m22=0):
    mat33 = np.array([[m00, m01, m02],
                      [m10, m11, m12],
                      [m20, m21, m22]])
    return mat33

#function to create a 3x3 Matrix from a 3d Vector

def matrix33fromvector3(vector3):
    mat33 = matrix33(m00= vector3[0,0],m11= vector3[0,1], m22= vector3[0,2])
    return mat33

# functions from cMatrix.cpp
#function to create a 4x4 Matrix

def matrix44(m00=0, m01=0, m02=0, m03=0,
             m10=0, m11=0, m12=0, m13=0,
             m20=0, m21=0, m22=0, m23=0,
             m30=0, m31=0, m32=0,m33=0):
    mat44 = np.array([[m00, m01, m02, m03],
                      [m10, m11, m12, m13],
                      [m20, m21, m22, m23],
                      [m30, m31, m32, m33]])
    return mat44

#function to create a 4x4 Matrix from a 3d Vector

def matrix44fromvector3(vector3):
    mat44 = matrix44(m03= vector3[0,0], m13= vector3[0,1], m23= vector3[0,2])
    return mat44

#function to create a 4x4 Matrix from a 3x3 Matrix

def matrix44frommatrix33(matrix33):
    mat44 = matrix44(m00= matrix33[0,0], m01= matrix33[0,1], m02= matrix33[0,2],
                     m10= matrix33[1,0], m11= matrix33[1,1], m12= matrix33[1,2],
                     m20= matrix33[2,0], m21= matrix33[2,1], m22= matrix33[2,2])
    return mat44

#function to create a 4x4 Matrix from a 3x3 Matrix and a 3d Vector

def matrix44frommatrix33andvector3(matrix33, vector3):
    mat44 = matrix44(m00= matrix33[0,0], m01= matrix33[0,1], m02= matrix33[0,2], m03= vector3[0,0],
                     m10= matrix33[1,0], m11= matrix33[1,1], m12= matrix33[1,2], m13= vector3[0,1],
                     m20= matrix33[2,0], m21= matrix33[2,1], m22= matrix33[2,2], m23= vector3[0,2])
    return mat44

#make a 4x4 Identity Matrix

def setIdentity():
    mat44 = matrix44(m00=1, m11=1, m22=1, m33=1)

#function to setDihedral

def setDihedral(phi, psi, R):
    mat44 = matrix44(m00=m.cos(psi), m01=m.sin(phi) * m.sin(psi), m02=m.cos(phi) * m.sin(psi), m03=R * m.cos(psi),
                     m10=0,m11=m.cos(phi), m12=m.sin(phi), m13=0,
                     m20=m.sin(psi), m21=m.sin(phi) * m.cos(psi), m22=m.cos(phi) * m.cos(psi), m23=R * m.sin(psi),
                     m30=0, m31=0, m32=0, m33=1)
    return mat44


#function to setDihedralDphi

def setDihedralDphi(phi, psi, R):
    mat44 = matrix44(m00=0, m01=m.cos(phi) * m.sin(psi), m02=m.sin(phi) * m.sin(psi), m03=0,
                     m10=0, m11=m.sin(phi),m12=m.cos(phi), m13=0,
                     m20=0, m21=m.cos(phi) * m.cos(psi), m22=m.sin(phi) * m.cos(psi), m23=0,
                     m30=0, m31=0, m32=0, m33=0)
    return mat44


#function to setDihedralDpsi

def setDihedralDpsi(phi, psi, R):
    mat44 = matrix44(m00=m.sin(psi), m01=m.sin(phi) * m.cos(psi), m02=m.cos(phi) * m.cos(psi), m03=R * m.sin(psi),
                     m10=0, m11=0, m12=0, m13=0,
                     m20=m.cos(psi), m21=m.sin(phi) * m.sin(psi), m22=m.cos(phi) * m.sin(psi), m23=R * m.cos(psi),
                     m30=0, m31=0, m32=0, m33=0)
    return mat44


#function to setDihedralDr

def setDihedralDr(phi, psi, R):
    mat44 = matrix44(m00=0, m01=0, m02=0, m03=m.cos(psi),
                     m10=0, m11=0, m12=0, m13=0,
                     m20=0, m21=0, m22=0,m23=-1 * m.sin(psi),
                     m30=0, m31=0, m32=0, m33=0)
    return mat44


#function to setRx

def setRx(angle):
    mat44 = matrix44(m00=1, m01=0, m02=0, m03=0,
                     m10=0, m11=m.cos(angle), m12=m.sin(angle), m13=0,
                     m20=0, m21=m.sin(angle), m22=m.cos(angle), m23=0,
                     m30=0, m31=0, m32=0, m33=1)
    return mat44


#function to setRy

def setRy(angle):
    mat44 = matrix44(m00=m.cos(angle), m01=0, m02=m.sin(angle), m03=0,
                     m10=0, m11=1, m12=0, m13=0,
                     m20=m.sin(angle), m21=0, m22=m.cos(angle), m23=0,
                     m30=0, m31=0, m32=0, m33=1)
    return mat44


#function to setRz

def setRz(angle):
    mat44 = matrix44(m00=m.cos(angle), m01=-1 * m.sin(angle), m02=0, m03=0,
                     m10=m.sin(angle), m11=m.cos(angle), m12=0, m13=0,
                     m20=0, m21=0, m22=1, m23=0,
                     m30=0, m31=0, m32=0, m33=1)
    return mat44


#function to setDRx

def setDRx(angle):
    mat44 = matrix44(m00=0, m01=0, m02=0, m03=0,
                     m10=0, m11=-1 * m.sin(angle), m12=-1 * m.cos(angle), m13=0,
                     m20=0, m21=m.cos(angle), m22=-1 * m.sin(angle), m23=0,
                     m30=0, m31=0, m32=0, m33=0)
    return mat44


#function to invertTransform44

def invertTransform44(mat44):
    Matrix33 = matrix33(1 / mat44[0, 0], mat44[0, 1], mat44[0, 2],
                        mat44[1, 0], 1 / mat44[1, 1], mat44[1, 2],
                        mat44[2, 0], mat44[2, 1], 1 / mat44[2, 2])

    Vector3 = vector3(mat44[0, 3], mat44[1, 3], mat44[2, 3])

    TranMat33 = np.transpose(Matrix33)

    invTransVec3 = np.transpose(Vector3) * (-1) * TranMat33

    invTransMat44 = matrix44(TranMat33[0, 0], TranMat33[0, 1], TranMat33[0, 2], invTransVec3[0, 0],
                             TranMat33[1, 0], TranMat33[1, 1], TranMat33[1, 2], invTransVec3[1, 1],
                             TranMat33[2, 0], TranMat33[2, 1], TranMat33[2, 2], invTransVec3[2, 2],
                             0, 0, 0, 1)

    return invTransMat44

#function to setT

def setT(mat44, translation, axis):
    if axis == "x":
        mat44[0, 3] = translation

    if axis == "y":
        mat44[1, 3] = translation

    if axis == "z":
        mat44[2, 3] = translation

    return mat44
