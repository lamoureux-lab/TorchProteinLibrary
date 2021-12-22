import math as m
import numpy as np
from Math.cMatrix import *


def updateMatrix(mat44, x, y,d, alpha, beta):
    '''
    :param x:
    :param y:
    :param d:
    :param alpha:
    :param beta:
    :return:
    '''
    Ry = setRy(beta)
    Tr = setT(mat44, d, "x")
    Rx = setRx(alpha)
    mat = Ry * Tr * Rx

    return mat

def updateDMatrix(mat44, x, y,d, alpha, beta):
    '''
    :param x:
    :param y:
    :param d:
    :param alpha:
    :param beta:
    :return: matrix for derivates wrt Rx
    '''
    Ry = setRy(beta)
    Tr = setT(mat44, d, "x")
    DRx = setDRx(alpha)
    dmat = Ry * Tr * DRx

    return dmat

def conformations(sequence, angle, angle_grad,  angles_length, atoms_global):
    '''
    :param sequence:
    :param angle:
    :param angle_grad:
    :param angles_length:
    :param atoms_global:
    :return:
    '''
    for i in range(len(sequence)):
        # T *phi = angles + i + angles_length*0;T *dphi = angles_grad + i + angles_length*0;
        #T *psi = angles + i + angles_length*1;T *dpsi = angles_grad + i + angles_length*1;

        #T * omega, *domega;
        #if (i > 0):
        #omega = angles + i-1 + angles_length * 2;domega = angles_grad + i-1 + angles_length * 2;
        #else:
        #omega = & geo.omega_const;domega = NULL;
        #// omega = & zero_const;domega = NULL;

        #T * xi1 = angles + i + angles_length * 3;T * dxi1 = angles_grad + i + angles_length * 3;
        #T * xi2 = angles + i + angles_length * 4;T * dxi2 = angles_grad + i + angles_length * 4;
        #T * xi3 = angles + i + angles_length * 5;T * dxi3 = angles_grad + i + angles_length * 5;
        #T * xi4 = angles + i + angles_length * 6;T * dxi4 = angles_grad + i + angles_length * 6;
        #T * xi5 = angles + i + angles_length * 7;T * dxi5 = angles_grad + i + angles_length * 7;
        #
        #

        #if (add_terminal):
        #    if (i == (len(sequence)-1)):
        #        terminal = true;
        #    else:
        #        terminal = false;
    ## cMatrix needs additional angles info
        pass


'''
Function for Graphing the transformation of the backbones. 
Represent each strand separately
'''
