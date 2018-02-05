from Exposed.cppVisualization import visSequence, updateAngles
import torch
import time
if __name__=='__main__':

    sequence = 'GG'
    angles_len = len(sequence)
    num_angles = 7
    angles = torch.DoubleTensor(num_angles, angles_len).zero_()

    visSequence(sequence)

    for i in range(0,100):
        
        for j in range(0,num_angles):
            for k in range(0, angles_len):
                angles[j,k]+=0.1
        
        time.sleep(1)
        updateAngles(angles)