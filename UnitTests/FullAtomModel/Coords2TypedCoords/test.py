import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import Angles2Coords, Coords2TypedCoords


class TestCoords2TypedCoords(unittest.TestCase):

    def setUp(self):
        self.sequence = ['GGGGGG', 'GGGGGG', 'QQQQQQ', 'MMMMMM', 'HHHHHH']
        angles = torch.zeros(len(self.sequence), 7, len(self.sequence[0]), dtype=torch.double, device='cpu')
        angles[0, 0, :] = -1.047
        angles[0, 1, :] = -0.698
        angles[0, 2:, :] = np.pi
        angles[0, 3:, :] = 110.4*np.pi/180.0
        a2c = Angles2Coords()
        self.coords, _, self.res_names, _, self.atom_names, self.num_atoms = a2c(angles, self.sequence)

        self.c2tc = Coords2TypedCoords()          # Default atom types - 11
        self.c2tcElement = Coords2TypedCoords(4)  # Element atom types - 4  (C,N,O,S)
        self.c2tcCharmm = Coords2TypedCoords(27)  # Charmm  atom types - 38 


class TestCoords2TypedCoordsForward(TestCoords2TypedCoords):
    def runTest(self):
        tcoords, num_atoms_of_type = self.c2tc(self.coords, self.res_names, self.atom_names, self.num_atoms)
        self.assertEqual(num_atoms_of_type[0, 0].item(), 0)  # sulfur
        self.assertEqual(num_atoms_of_type[0, 1].item(), len(self.sequence[0]))  # nitrogen amide
        self.assertEqual(num_atoms_of_type[0, 2].item(), 0)  # nitrogen arom
        self.assertEqual(num_atoms_of_type[0, 3].item(), 0)  # nitrogen guan
        self.assertEqual(num_atoms_of_type[0, 4].item(), 0)  # nitrogen ammon
        self.assertEqual(num_atoms_of_type[0, 5].item(), len(self.sequence[0]))  # oxygen carbonyl
        self.assertEqual(num_atoms_of_type[0, 6].item(), 0)  # oxygen hydroxyl
        self.assertEqual(num_atoms_of_type[0, 7].item(), 0)  # oxygen carboxyl
        self.assertEqual(num_atoms_of_type[0, 8].item(), len(self.sequence[0]))  # oxygen carboxyl
        self.assertEqual(num_atoms_of_type[0, 9].item(), 0)  # carbon sp2
        self.assertEqual(num_atoms_of_type[0, 10].item(), len(self.sequence[0]))  # carbon sp3

        for i in range(1, 2):  # batch check
            for j in range(11):
                self.assertEqual(num_atoms_of_type[i, j], num_atoms_of_type[i-1, j])


class TestCoords2TypedCoordsElementForward(TestCoords2TypedCoords):
    def runTest(self):
        tcoords, num_atoms_of_type = self.c2tcElement(self.coords, self.res_names, self.atom_names, self.num_atoms)
        self.assertEqual(num_atoms_of_type[0, 0].item(), 12) # C 
        self.assertEqual(num_atoms_of_type[0, 1].item(), 6)  # N
        self.assertEqual(num_atoms_of_type[0, 2].item(), 6)  # O
        self.assertEqual(num_atoms_of_type[0, 3].item(), 0)  # S
      


class TestCoords2TypedCoordsCharmmForward(TestCoords2TypedCoords):
    def runTest(self):
        tcoords, num_atoms_of_type = self.c2tcCharmm(self.coords, self.res_names, self.atom_names, self.num_atoms)

        n_types = 27
        # Testing GLY
        i_num_types = 0
        for i in range(n_types):
            j_num_types = num_atoms_of_type[1, i].item()
            if i == 0:             # C
                i_num_types = 6
            elif i == 4:           # CT2  
                i_num_types = 6
            elif i == 18:          # NH1
                i_num_types = 6
            elif i == 23:          # O
                i_num_types = 6
            else:
                i_num_types = 0
                     
            self.assertEqual(j_num_types, i_num_types)      
            
            
        # Testing GLN
        i_num_types = 0
        for i in range(n_types):
            j_num_types = num_atoms_of_type[2, i].item()
            if i == 0:             # C
                i_num_types = 6
            elif i == 3:           # CT1
                i_num_types = 6    
            elif i == 4:           # CT2  
                i_num_types = 12
            elif i == 14:          # CC
                i_num_types = 6
            elif i == 18:          # NH1
                i_num_types = 6
            elif i == 19:          # NH2
                i_num_types = 6
            elif i == 23:          # O  
                i_num_types = 12
            else:
                i_num_types = 0
                     
            self.assertEqual(j_num_types, i_num_types)      

            
        # Testing MET
        i_num_types = 0
        for i in range(n_types):
            j_num_types = num_atoms_of_type[3, i].item()
            if i == 0:             # C
                i_num_types = 6
            elif i == 3:           # CT1
                i_num_types = 6    
            elif i == 4:           # CT2  
                i_num_types = 12
            elif i == 6:           # CT3
                i_num_types = 6
            elif i == 18:          # NH1
                i_num_types = 6
            elif i == 23:          # O  
                i_num_types = 6
            elif i == 26:          # S  
                i_num_types = 6
            else:
                i_num_types = 0
                     
            self.assertEqual(j_num_types, i_num_types)      

        # Testing HIS
        i_num_types = 0
        for i in range(n_types):
            j_num_types = num_atoms_of_type[4, i].item()
            if i == 0:             # C
                i_num_types = 6
            elif i == 3:             # CT1
                i_num_types = 6
            elif i == 4:           # CT2  
                i_num_types = 6
            elif i == 7:             # CPH1
                i_num_types = 12
            elif i == 8:             # CPH2
                i_num_types = 6
            elif i == 17:            # NR
                i_num_types = 12   
            elif i == 18:          # NH1
                i_num_types = 6
            elif i == 23:          # O
                i_num_types = 6
            else:
                i_num_types = 0
                     
            self.assertEqual(j_num_types, i_num_types)      

            
                
class TestCoords2TypedCoordsBackward(TestCoords2TypedCoords):
    def runTest(self):
        self.coords.requires_grad_()
        tcoords, num_atoms_of_type = self.c2tc(self.coords, self.res_names, self.atom_names, self.num_atoms)
        z0 = tcoords.sum()
        z0.backward()
        back_grad_x0 = torch.zeros_like(self.coords).copy_(self.coords.grad)
        error = 0.0
        N = 0
        x1 = torch.zeros_like(self.coords)
        for i in range(0, self.coords.size(0)):
            for j in range(0, self.coords.size(1)):
                dx = 0.01
                x1.copy_(self.coords)
                x1[i, j] += dx
                x1coords, num_atoms_of_type = self.c2tc(x1, self.res_names, self.atom_names, self.num_atoms)
                z1 = x1coords.sum()
                dy_dx = (z1.item()-z0.item())/(dx)
                error += torch.abs(dy_dx - back_grad_x0[i, j]).item()
                N += 1

        error /= float(N)
        self.assertLess(error, 1E-5)


class TestCoords2TypedCoordsElementBackward(TestCoords2TypedCoords):
    def runTest(self):
        self.coords.requires_grad_()
        tcoords, num_atoms_of_type = self.c2tcElement(self.coords, self.res_names, self.atom_names, self.num_atoms)
        z0 = tcoords.sum()
        z0.backward()
        back_grad_x0 = torch.zeros_like(self.coords).copy_(self.coords.grad)
        error = 0.0
        N = 0
        x1 = torch.zeros_like(self.coords)
        for i in range(0, self.coords.size(0)):
            for j in range(0, self.coords.size(1)):
                dx = 0.01
                x1.copy_(self.coords)
                x1[i, j] += dx
                x1coords, num_atoms_of_type = self.c2tc(x1, self.res_names, self.atom_names, self.num_atoms)
                z1 = x1coords.sum()
                dy_dx = (z1.item()-z0.item())/(dx)
                error += torch.abs(dy_dx - back_grad_x0[i, j]).item()
                N += 1

        error /= float(N)
        self.assertLess(error, 1E-5)


class TestCoords2TypedCoordsCharmmBackward(TestCoords2TypedCoords):
    def runTest(self):
        self.coords.requires_grad_()
        tcoords, num_atoms_of_type = self.c2tcCharmm(self.coords, self.res_names, self.atom_names, self.num_atoms)
        z0 = tcoords.sum()
        z0.backward()
        back_grad_x0 = torch.zeros_like(self.coords).copy_(self.coords.grad)
        error = 0.0
        N = 0
        x1 = torch.zeros_like(self.coords)
        for i in range(0, self.coords.size(0)):
            for j in range(0, self.coords.size(1)):
                dx = 0.01
                x1.copy_(self.coords)
                x1[i, j] += dx
                x1coords, num_atoms_of_type = self.c2tc(x1, self.res_names, self.atom_names, self.num_atoms)
                z1 = x1coords.sum()
                dy_dx = (z1.item()-z0.item())/(dx)
                error += torch.abs(dy_dx - back_grad_x0[i, j]).item()
                N += 1

        error /= float(N)
        self.assertLess(error, 1E-5)
        
        
if __name__ == "__main__":
    unittest.main()
