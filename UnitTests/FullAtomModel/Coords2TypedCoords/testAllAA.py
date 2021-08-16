import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import Angles2Coords, Coords2TypedCoords


class TestCoords2TypedCoords(unittest.TestCase):

    def setUp(self):
        
        d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
             'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
        n_aa = 6
        Alist = list(d.values())
        for i in range(len(Alist)):
            Alist[i] = Alist[i]*n_aa
        
        self.sequence = Alist
        
        angles = torch.zeros(len(self.sequence), 7, len(self.sequence[0]), dtype=torch.double, device='cpu')
        angles[0, 0, :] = -1.047
        angles[0, 1, :] = -0.698
        angles[0, 2:, :] = np.pi
        angles[0, 3:, :] = 110.4*np.pi/180.0
        a2c = Angles2Coords()
        self.coords, _, self.res_names, _, self.atom_names, self.num_atoms = a2c(angles, self.sequence)

        #self.c2tc = Coords2TypedCoords()          # Default atom types - 11
        #self.c2tcElement = Coords2TypedCoords(4)  # Element atom types - 4  (C,N,O,S)
        self.c2tcCharmm = Coords2TypedCoords(26)  # Charmm  atom types - 26 


class TestCoords2TypedCoordsCharmmForward(TestCoords2TypedCoords):
    def runTest(self):
        tcoords, num_atoms_of_type = self.c2tcCharmm(self.coords, self.res_names, self.atom_names, self.num_atoms)

        for i_seq, seq in enumerate(self.sequence):

            n_types = 26
            if seq[0] == 'A':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1  
                        i_num_types = 6
                    elif i == 5:           # CT31
                        i_num_types = 6
                    elif i == 17:          # NH1  
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item() , i_num_types)      

            if seq[0] == 'R':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 12
                    elif i == 2:           # CT1
                        i_num_types = 6
                    elif i == 3:           # CT2  
                        i_num_types = 18
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 20:          # NC2  
                        i_num_types = 18
                    elif i == 22:          # O  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)

            if seq[0] == 'N':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1
                        i_num_types = 6    
                    elif i == 3:           # CT2  
                        i_num_types = 6
                    elif i == 13:          # CC  
                        i_num_types = 6
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 18:          # NH2  
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 12
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)

            if seq[0] == 'D':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1
                        i_num_types = 6    
                    elif i == 4:           # CT2A  
                        i_num_types = 6
                    elif i == 13:          # CC  
                        i_num_types = 6
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    elif i == 23:          # OC  
                        i_num_types = 12
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)
   
            if seq[0] == 'C':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1
                        i_num_types = 6
                    elif i == 3:           # CT2  
                        i_num_types = 6
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    elif i == 25:          # S  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)

            if seq[0] == 'Q':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1
                        i_num_types = 6    
                    elif i == 3:           # CT2  
                        i_num_types = 12
                    elif i == 13:          # CC
                        i_num_types = 6
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 18:          # NH2
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 12
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)      

            if seq[0] == 'E':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1
                        i_num_types = 6    
                    elif i == 3:           # CT2  
                        i_num_types = 6
                    elif i == 4:           # CT2A  
                        i_num_types = 6
                    elif i == 13:          # CC
                        i_num_types = 6
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    elif i == 23:          # OC  
                        i_num_types = 12
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)  
                    
            if seq[0] == 'G':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 3:           # CT2  
                        i_num_types = 6
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)

            if seq[0] == 'H':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1
                        i_num_types = 6
                    elif i == 3:           # CT2  
                        i_num_types = 6
                    elif i == 6:           # CPH1
                        i_num_types = 12
                    elif i == 7:           # CPH2
                        i_num_types = 6
                    elif i == 16:          # NR
                        i_num_types = 12
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    #print(i)    
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)

            if seq[0] == 'I':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1  
                        i_num_types = 12
                    elif i == 3:           # CT2  
                        i_num_types = 6
                    elif i == 5:           # CT3  
                        i_num_types = 12
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)      

            if seq[0] == 'L':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1  
                        i_num_types = 12
                    elif i == 3:           # CT2  
                        i_num_types = 6
                    elif i == 5:           # CT3  
                        i_num_types = 12
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)      

            if seq[0] == 'K':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1  
                        i_num_types = 6
                    elif i == 3:           # CT2  
                        i_num_types = 24
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 19:          # NH3
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)      
        
            if seq[0] == 'M':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1
                        i_num_types = 6    
                    elif i == 3:           # CT2  
                        i_num_types = 12
                    elif i == 5:           # CT3
                        i_num_types = 6
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    elif i == 25:          # S  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)      

            if seq[0] == 'F':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 1:           # CA
                        i_num_types = 36
                    elif i == 2:           # CT1
                        i_num_types = 6    
                    elif i == 3:           # CT2  
                        i_num_types = 6
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)      

            if seq[0] == 'P':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 10:          # CP1
                        i_num_types = 6
                    elif i == 11:          # CP2
                        i_num_types = 12    
                    elif i == 12:          # CP3  
                        i_num_types = 6
                    elif i == 15:          # N
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types) 

            if seq[0] == 'S':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1
                        i_num_types = 6    
                    elif i == 3:           # CT2  
                        i_num_types = 6
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    elif i == 24:          # OH1  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)      
                    
            if seq[0] == 'T':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1
                        i_num_types = 12    
                    elif i == 5:           # CT3
                        i_num_types = 6
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    elif i == 24:          # OH1  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)      

            if seq[0] == 'W':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 1:           # CA
                        i_num_types = 18
                    elif i == 2:           # CT1
                        i_num_types = 6    
                    elif i == 3:           # CT2
                        i_num_types = 6
                    elif i == 8:           # CPT
                        i_num_types = 12
                    elif i == 9:          # CY
                        i_num_types = 6
                    elif i == 14:          # CAI
                        i_num_types = 12
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 21:          # NY
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i].item(), i_num_types)      
                    
            if seq[0] == 'Y':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 1:           # CA
                        i_num_types = 36
                    elif i == 2:           # CT1  
                        i_num_types = 6
                    elif i == 3:           # CT2  
                        i_num_types = 6
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    elif i == 24:          # OH1  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i], i_num_types)

            if seq[0] == 'V':
                print('Testing ' + seq)
                for i in range(n_types):
                    if i == 0:             # C
                        i_num_types = 6
                    elif i == 2:           # CT1  
                        i_num_types = 12
                    elif i == 5:           # CT3  
                        i_num_types = 12
                    elif i == 17:          # NH1
                        i_num_types = 6
                    elif i == 22:          # O  
                        i_num_types = 6
                    else:
                        i_num_types = 0
                    self.assertEqual(num_atoms_of_type[i_seq, i], i_num_types)
                    
if __name__ == "__main__":
    unittest.main()
