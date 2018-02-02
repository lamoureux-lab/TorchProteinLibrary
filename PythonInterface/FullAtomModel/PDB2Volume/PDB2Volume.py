import numpy as np

import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module

from Exposed import cppPDB2Volume

def convertStringList(stringList):
    '''Converts list of strings to 0-terminated byte tensor'''
    maxlen = 0
    for string in stringList:
        string += '\0'
        if len(string)>maxlen:
            maxlen = len(string)
    ar = np.zeros( (len(stringList), maxlen), dtype=np.uint8)
    
    for i,string in enumerate(stringList):
        npstring = np.fromstring(string, dtype=np.uint8)
        ar[i,:npstring.shape[0]] = npstring
    
    return torch.from_numpy(ar)

def convertString(string):
    '''Converts a string to 0-terminated byte tensor'''  
    return torch.from_numpy(np.fromstring(string+'\0', dtype=np.uint8))

class PDB2Volume:
    def __init__(self):
        pass

    def __call__(self, stringList):

        if type(stringList)==str:
            stringListTensor = convertString(stringList)
        elif type(stringList)==list:
            if len(stringList)==1:
                stringListTensor = convertString(stringList[0])
            else:
                stringListTensor = convertStringList(stringList)
        else:
            raise Exception("Unknown input format:", stringList)

        cppPDB2Volume.PDB2Volume(stringListTensor)
