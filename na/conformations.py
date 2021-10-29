import math as m
import numpy as np
from cMatrix import *


def updateMatrix(x, y,d, alpha, beta):
    '''
    :param x:
    :param y:
    :param d:
    :param alpha:
    :param beta:
    :return:
    '''
    Ry = setRy(beta)
    Rx = setRx(alpha)
    mat = Ry * d * Rx

def updateDMatrix(x, y,d, alpha, beta):
    '''
    :param x:
    :param y:
    :param d:
    :param alpha:
    :param beta:
    :return: matrix for derivates wrt Rx
    '''
    Ry = setRy(beta)
    DRx = setDRx(alpha)
    mat = Ry * d * DRx

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

    ## cMatrix needs additional angles info
        pass


'''
Function for Graphing the transformation of the backbones. 
Represent each strand separately
'''
