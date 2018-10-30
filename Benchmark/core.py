import torch
from torch.autograd import profiler
from TorchProteinLibrary import ReducedModel, FullAtomModel, RMSD
import numpy as np

class ModuleBenchmark:
    def __init__(self, device='cpu'):
        self.module = None

    def prepare(self):
        pass
    
    def run_forward(self):
        pass

    def run_backward(self):
        pass

    def measure_forward(self):
        self.prepare()
        with profiler.profile() as prof:
            self.run_forward()
        dur = 0.0
        for evt in prof.function_events:
            dur += evt.cpu_interval.elapsed_us()
        return dur


    def measure_backward(self):        
        self.prepare()
        with profiler.profile() as prof:
            self.run_backward()
        dur = 0.0
        for evt in prof.function_events:
            dur += evt.cpu_interval.elapsed_us()
        return dur
        
    