from PDB2Volume import PDB2Volume

if __name__=='__main__':
    a = ["1 hello world", "2 hello world", "3 hello world"]
    pdb2v = PDB2Volume()
    pdb2v(a)
    pdb2v(["fuck off"])
    pdb2v("fuck off")
    