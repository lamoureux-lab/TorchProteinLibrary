import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import Angles2Coords, Coords2TypedCoords


class TestCoords2TypedCoords(unittest.TestCase):

    def setUp(self):
        self.sequence = ['GGGGGG', 'GGGGGG']
        angles = torch.zeros(len(self.sequence), 7, len(self.sequence[0]), dtype=torch.double, device='cpu')
        angles[0, 0, :] = -1.047
        angles[0, 1, :] = -0.698
        angles[0, 2:, :] = np.pi
        angles[0, 3:, :] = 110.4*np.pi/180.0
        a2c = Angles2Coords()
        self.coords, _, self.res_names, _, self.atom_names, self.num_atoms = a2c(angles, self.sequence)
        self.c2tc = Coords2TypedCoords()          # Default atom types - 11
        self.c2tcElement = Coords2TypedCoords(4)  # Element atom types - 4  (C,N,O,S)
        self.c2tcCharmm = Coords2TypedCoords(38)  # Charmm  atom types - 38 


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

        for i in range(1, len(self.sequence)):  # batch check
            for j in range(11):
                self.assertEqual(
                    num_atoms_of_type[i, j], num_atoms_of_type[i-1, j])


class TestCoords2TypedCoordsElementForward(TestCoords2TypedCoords):
    def runTest(self):
        tcoords, num_atoms_of_type = self.c2tcElement(self.coords, self.res_names, self.atom_names, self.num_atoms)
        self.assertEqual(num_atoms_of_type[0, 0].item(), 12)  # C 
        self.assertEqual(num_atoms_of_type[0, 1].item(), len(self.sequence[0]))  # N
        self.assertEqual(num_atoms_of_type[0, 2].item(), 6)  # O
        self.assertEqual(num_atoms_of_type[0, 3].item(), 0)  # S

        for i in range(1, len(self.sequence)):  # batch check
            for j in range(4):
                self.assertEqual(num_atoms_of_type[i, j], num_atoms_of_type[i-1, j])


class TestCoords2TypedCoordsCharmmForward(TestCoords2TypedCoords):
    def runTest(self):
        tcoords, num_atoms_of_type = self.c2tcCharmm(self.coords, self.res_names, self.atom_names, self.num_atoms)
        self.assertEqual(num_atoms_of_type[0, 0].item(), 6)   # C    - carbonyl C, peptide backbone		  
        self.assertEqual(num_atoms_of_type[0, 1].item(), 0)   # CA   - aromatic C				  
        self.assertEqual(num_atoms_of_type[0, 2].item(), 0)   # CT   - aliphatic sp3 C, new LJ params, no hydrogen
        self.assertEqual(num_atoms_of_type[0, 3].item(), 0)   # CT1  - aliphatic sp3 C for CH			  
        self.assertEqual(num_atoms_of_type[0, 4].item(), 6)   # CT2  - aliphatic sp3 C for CH2			  
        self.assertEqual(num_atoms_of_type[0, 5].item(), 0)   # CT2A - from CT2 (asp, glu, hsp chi1/chi2 fitting) 
        self.assertEqual(num_atoms_of_type[0, 6].item(), 0)   # CT3  - aliphatic sp3 C for CH3			  
        self.assertEqual(num_atoms_of_type[0, 7].item(), 0)   # CPH1 - his CG and CD2 carbons			  
        self.assertEqual(num_atoms_of_type[0, 8].item(), 0)   # CPH2 - his CE1 carbon				  
        self.assertEqual(num_atoms_of_type[0, 9].item(), 0)   # CPT  - trp C between rings			  
        self.assertEqual(num_atoms_of_type[0, 10].item(), 0)  # CY   - TRP C in pyrrole ring			  
        self.assertEqual(num_atoms_of_type[0, 11].item(), 0)  # CP1  - tetrahedral C (proline CA)		  
        self.assertEqual(num_atoms_of_type[0, 12].item(), 0)  # CP2  - tetrahedral C (proline CB/CG)		  
        self.assertEqual(num_atoms_of_type[0, 13].item(), 0)  # CP3  - tetrahedral C (proline CD)		  
        self.assertEqual(num_atoms_of_type[0, 14].item(), 0)  # CC   - carbonyl C, asn,asp,gln,glu,cter,ct2	  
        self.assertEqual(num_atoms_of_type[0, 15].item(), 0)  # CD   - carbonyl C, pres aspp,glup,ct1		  
        self.assertEqual(num_atoms_of_type[0, 16].item(), 0)  # CS   - thiolate carbon				  
        self.assertEqual(num_atoms_of_type[0, 17].item(), 0)  # CE1  - for alkene; RHC=CR			  
        self.assertEqual(num_atoms_of_type[0, 18].item(), 0)  # CE2  - for alkene; H2C=CR			  
        self.assertEqual(num_atoms_of_type[0, 19].item(), 0)  # CAI  - aromatic C next to CPT in trp		  
        self.assertEqual(num_atoms_of_type[0, 20].item(), 0)  # N    - proline N				  
        self.assertEqual(num_atoms_of_type[0, 21].item(), 0)  # NR1  - neutral his protonated ring nitrogen	  
        self.assertEqual(num_atoms_of_type[0, 22].item(), 0)  # NR2  - neutral his unprotonated ring nitrogen	  
        self.assertEqual(num_atoms_of_type[0, 23].item(), 0)  # NR3  - charged his ring nitrogen		  
        self.assertEqual(num_atoms_of_type[0, 24].item(), 6)  # NH1  - peptide nitrogen				          
        self.assertEqual(num_atoms_of_type[0, 25].item(), 0)  # NH2  - amide nitrogen				  
        self.assertEqual(num_atoms_of_type[0, 26].item(), 0)  # NH3  - ammonium nitrogen			  
        self.assertEqual(num_atoms_of_type[0, 27].item(), 0)  # NC2  - guanidinium nitrogen			  
        self.assertEqual(num_atoms_of_type[0, 28].item(), 0)  # NY   - TRP N in pyrrole ring			  
        self.assertEqual(num_atoms_of_type[0, 29].item(), 0)  # NP   - Proline ring NH2+ (N-terminal)		  
        self.assertEqual(num_atoms_of_type[0, 30].item(), 6)  # O    - carbonyl oxygen				  
        self.assertEqual(num_atoms_of_type[0, 31].item(), 0)  # OB   - carbonyl oxygen in acetic acid		  
        self.assertEqual(num_atoms_of_type[0, 32].item(), 0)  # OC   - carboxylate oxygen			  
        self.assertEqual(num_atoms_of_type[0, 33].item(), 0)  # OH1  - hydroxyl oxygen				  
        self.assertEqual(num_atoms_of_type[0, 34].item(), 0)  # OS   - ester oxygen				  
        self.assertEqual(num_atoms_of_type[0, 35].item(), 0)  # S    - sulphur					  
        self.assertEqual(num_atoms_of_type[0, 36].item(), 0)  # SM   - sulfur C-S-S-C type			  
        self.assertEqual(num_atoms_of_type[0, 37].item(), 0)  # SS   - thiolate sulfur                            
        
        for i in range(1, len(self.sequence)):  # batch check         
            for j in range(38):
                self.assertEqual(
                    num_atoms_of_type[i, j], num_atoms_of_type[i-1, j])


                
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


if __name__ == "__main__":
    unittest.main()
