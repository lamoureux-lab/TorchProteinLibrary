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
    def __init__(self, box_size=120, resolution=1.0, rotate=True, translate=True):
        self.box_size = box_size
        self.resolution = resolution
        self.num_atom_types = 11
        self.batch_size = None        
        self.volume = None
        self.use_cuda = False
        self.rotate = rotate
        self.translate = translate

    def cuda(self):
        self.use_cuda=True

    def __call__(self, stringList):

        if type(stringList)==str:
            stringListTensor = convertString(stringList)
            volume = torch.FloatTensor(self.num_atom_types, self.box_size, self.box_size, self.box_size)
        elif type(stringList)==list:
            if len(stringList)==1:
                stringListTensor = convertString(stringList[0])
                volume = torch.FloatTensor(self.num_atom_types, self.box_size, self.box_size, self.box_size)
            else:
                stringListTensor = convertStringList(stringList)
                self.batch_size = len(stringList)
                volume = torch.FloatTensor(self.batch_size, self.num_atom_types, self.box_size, self.box_size, self.box_size)
        else:
            raise Exception("Unknown input format:", stringList)

        if self.use_cuda:
            volume = volume.cuda()

        volume.fill_(0.0)
        if self.use_cuda:
            cppPDB2Volume.PDB2VolumeCUDA(stringListTensor, volume, self.rotate, self.translate)
        else:    
            cppPDB2Volume.PDB2Volume(stringListTensor, volume)

        return volume


class PDB2VolumeLocal:
    def __init__(self, box_size=120, resolution=1.0, rotate=True, translate=True):
        self.box_size = box_size
        self.resolution = resolution
        self.num_atom_types = 11
                
        self.rotate = rotate
        self.translate = translate
        
    def cuda(self):
        self.use_cuda=True

    def __call__(self, stringList):

        if type(stringList)==list:
            stringListTensor = convertStringList(stringList)
            batch_size = len(stringList)
            volume = torch.FloatTensor(batch_size, self.num_atom_types, self.box_size, self.box_size, self.box_size).cuda()
            coords = torch.FloatTensor(batch_size, 1).cuda()
            num_atoms = torch.IntTensor(batch_size).cuda()
        else:
            raise Exception("Unknown input format:", stringList)
        
        volume.fill_(0.0)
        coords.fill_(0.0)
        num_atoms.fill_(0)
        cppPDB2Volume.PDB2VolumeLocal(stringListTensor, volume, coords, num_atoms, self.rotate, self.translate)

        return volume, coords, num_atoms
