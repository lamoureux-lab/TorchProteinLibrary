import gym
import torch
import random
import math
from ABModel import env

def move_naive(angles):
    angles.fill_(0.0)
    angles_length = angles.size(1)
    angle_index = random.randint(0, angles_length-1)
    angles[:,angle_index] = 2.0*math.pi*(torch.rand(2).cuda()-1.0)

if __name__=='__main__':
    env = gym.make('ABModel-v0')

    move = torch.FloatTensor(env.angles.size()).cuda()
    done = False
    while not done:
        move_naive(move)
        obs, rew, done, _ = env.step(move)
        if rew==1.0:
            print env.prev_energy

    env.render()