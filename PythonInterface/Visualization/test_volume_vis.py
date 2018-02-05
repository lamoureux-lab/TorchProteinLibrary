from Exposed.cppVisualization import VisualizeVolume4d
import torch
import time
import numpy as np
if __name__=='__main__':

    volume = torch.FloatTensor(11,100,100,100).fill_(0.0)

    for i in range(0,100):
        for j in range(0,100):
            for k in range(0, 100):
                r = np.sqrt(((i-50.0)*(i-50.0) + (j-50.0)*(j-50.0)+(k-50.0)*(k-50.0)))
                f = 30.0/(r+0.1)
                volume[0,i,j,k] = f
        
    VisualizeVolume4d(volume)