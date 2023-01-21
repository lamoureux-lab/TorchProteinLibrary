## function to read in PDB files and extract atom_name, res_names,chain_names,res_nums
from Bio.PDB import *
import barnaba as bb
#from barnaba import definitions

# file = '/u2/home_u2/fam95/Documents/1ffk.pdb'
file = '/u2/home_u2/fam95/Documents/119d.pdb'

class PDBLoader:
    '''
    PDB reader class, get atom_name, residues, chain_names etc.
    '''
    parser = PDBParser()
    struct = parser.get_structure('na', file)
    residues = []
    res_seq = [] # residue sequence number
    atoms = []
    res_names = []
    chain_name = []
    coords = [] # (x,y,z coordinates)

    for model in struct:
        for chain in model:
            chain_name.append(chain) ## get chain info
            for residue in chain:
                residues.append(residue) ## get residue info
                res_seq.append(residue.get_full_id()[3][1])
                for atom in residue:
                    atoms.append(atom)
                    x, y, z = atom.get_coord() ## get xyz coordinates
                    coords.append((x, y, z))
    # print(residues)
    #io = PDBIO()
    #io.set_structure(struct)
    #io.save("out.pdb")

    def NAfromPDB(struct):

        nucleotides = ["DA", "DC", "DG", "DT"]  # , "DU"]
        residues = []
        res_seq = []  # residue sequence number
        atoms = []
        res_names = []
        chain_name = []
        coords = []  # (x,y,z coordinates)

        for model in struct:
            for chain in model:
                chain_name.append(chain)  ## get chain info
                for residue in chain:
                    res_name = residue.get_resname()
                    strip_resname = res_name.strip()
                    if strip_resname in nucleotides:
                        residues.append(residue)  ## get residue info
                        res_seq.append(residue.get_full_id()[3][1])
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
        io.save("out.pdb")


    def getNAAngles(file):
        angles, res = bb.backbone_angles(file)
        #header = "# Residue " + "".join(["%10s " % aa for aa in definitions.bb_angles])
        #print(header)
        for j in range(angles.shape[1]):
            stri = "%10s" % res[j]
            for k in range(angles.shape[2]):
                stri += "%10.3f " % angles[0, j, k]
            print(stri)

    def getNAring_ang(file):
        ring_ang, resid = bb.sugar_angles(file)
        #header = "# Residue " + "".join(["%10s " % aa for aa in definitions.sugar_angles])
        #print(header)
        for j in range(ring_ang.shape[1]):
            stri = "%10s" % resid[j]
            for k in range(ring_ang.shape[2]):
                stri += "%10.3f " % ring_ang[0, j, k]
            print(stri)

#checking doc
# print(PDBLoader.__doc__)