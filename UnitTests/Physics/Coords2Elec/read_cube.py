import numpy as np

def cube2numpy(file_path):
    """
    
    """
    
    file_in = file_path
    inp = open(file_in,'r')
    cube = inp.readlines()
    inp.close()

    rB = 0.529177 #Bohr radius
    res = float(cube[3].split()[1])*rB

    
    #get spacial dimention, num of cells in 1D
    nx = int(cube[3].split()[0])
    ny = int(cube[4].split()[0])
    nz = int(cube[5].split()[0]) 

    #initilize empty arrays for flat and 3D phi
    Phi_flat = np.zeros(nx*ny*nz)
    Phi = np.zeros((ny,nx,nz))

    #unpack .cube starting from line7 to get phi values
    p = []
    for iline in cube[7:]:
        list_phi = iline.split()
        
        for el in list_phi:
            p.append(el)
            
    Phi_flat = np.array(p, dtype=float)
    Phi = Phi_flat.reshape(nx,ny,nz)
    
    return Phi, nx, res
