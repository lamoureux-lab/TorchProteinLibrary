## function to read in PDB files and extract atom_name, res_names,chain_names,res_nums

from Bio.PDB import *

file = '/home/anushriya/Documents/Lamoureux_lab/TorchProteinLibrary/na/1ffk.pdb'

class PDBLoader:
    '''
    PDB reader class, get atom_name, residues, chain_names etc.
    '''
    parser = PDBParser()
    struct = parser.get_structure('na', file)
    residues = []
    resseq = [] # residue sequence number
    atoms = []
    res_names = []
    chain_name = []
    coords = [] # (x,y,z coordinates)

    for model in struct:
        for chain in model:
            chain_name.append(chain) ## get chain info
            for residue in chain:
                residues.append(residue) ## get residue info
                resseq.append(residue.get_full_id()[3][1])
                for atom in residue:
                    atoms.append(atom)
                    x, y, z = atom.get_coord() ## get xyz coordinates
                    coords.append((x, y, z))



#checking doc
# print(PDBLoader.__doc__)