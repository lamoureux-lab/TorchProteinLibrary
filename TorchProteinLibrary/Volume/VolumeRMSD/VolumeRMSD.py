import torch
import torch.nn.functional as F
from torch.autograd import Function
import math

import sys
import os

import _Volume

def VolumeRMSD(coords, num_atoms, R0, R1, T0, resolution, volume_size):
    batch_size = coords.size(0)
    rmsdVolume = torch.zeros(batch_size, volume_size, volume_size, volume_size, dtype=torch.float, device='cuda')
    _Volume.VolumeGenRMSD(coords, num_atoms, R0, R1, T0, rmsdVolume, resolution)
    return rmsdVolume
