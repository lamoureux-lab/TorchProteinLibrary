## function to read in PDB files and extract atom_name, res_names,chain_names,res_nums

from Bio.PDB import *
file = '/home/anushriya/Documents/Lamoureux_lab/TorchProteinLibrary/na/3nir.pdb'

# for line in open(file):
#     splitting = line.split()
#     # print(spliting)
#     id = splitting[0]
#     header =[]
#     if id != 'Atom':
#         header.append(id)
#         # print(header)
#     if id == 'ATOM':
#         atom_name = splitting[2]



class PDBLoader:
    '''
    PDB reader class, get atom_name, residues, chain_names etc.
    '''
    parser = PDBParser()
    struct = parser.get_structure('test', file)
    for atom in struct.get_atoms():
        # print(atom)
        pass

    for residue in struct.get_residues():
        print(residue)
        # pass


#checking doc
# print(PDBLoader.__doc__)