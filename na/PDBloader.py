## function to read in PDB files and extract atom_name, res_names,chain_names,res_nums
from Bio.PDB import *

file = '/u2/home_u2/fam95/Documents/1ffk.pdb'

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


    io = PDBIO()
    io.set_structure(struct)
    io.save("out.pdb")

    def NAfromPDB(struct):

        Nucleotides = ["A", "C", "G", "U", "I"," DA", " DC", " DG", " DT", "DU", "I", "N"]
        residues = []
        resseq = []  # residue sequence number
        atoms = []
        res_names = []
        chain_name = []
        coords = []  # (x,y,z coordinates)

        for model in struct:
            for chain in model:
                chain_name.append(chain)  ## get chain info
                for residue in chain:
                    if residue.get_resname() in Nucleotides:
                        residues.append(residue)  ## get residue info
                        resseq.append(residue.get_full_id()[3][1])
                        for atom in residue:
                            atoms.append(atom)
                            x, y, z = atom.get_coord()  ## get xyz coordinates
                            coords.append((x, y, z))

                    else:
                        res2del = residue.get_id()
                        chain.detach_child(res2del)
                        continue


        io = PDBIO()
        io.set_structure(struct)
        io.save("out2.pdb")


#checking doc
# print(PDBLoader.__doc__)