#include <Kernels.h>


__global__ void projectToTensor(double* coords, uint* num_atoms_of_type, uint* offsets, float *volume, 
                                uint spatial_dim, float res){
	/*
    Input:
        coords: coordinates in a flat array:
            coords: {protein1, ... proteinN}
            protein1: {atom_type1 .. atom_typeM}
            atom_type: {x1,y1,z1 .. xL,yL,zL}
        num_atoms_of_type: number of atoms in each atom_type 
        offsets: offset for coordinates for each atom_type volume
    Output: 
        volume: density
    */
		int d = 2;
		size_t func_index = threadIdx.x + blockIdx.x*blockDim.x;
		float *type_volume = volume + func_index * spatial_dim*spatial_dim*spatial_dim;
		double *atoms_coords = coords + offsets[func_index];
		uint n_atoms = num_atoms_of_type[func_index];
		for(int atom_idx = 0; atom_idx<n_atoms; atom_idx+=3){
			double 	x = atoms_coords[atom_idx],
					y = atoms_coords[atom_idx + 1],
					z = atoms_coords[atom_idx + 2];
			int x_i = floor(x/res);
			int y_i = floor(y/res);
			int z_i = floor(z/res);
			for(int i=x_i-d; i<=(x_i+d);i++){
				for(int j=y_i-d; j<=(y_i+d);j++){
					for(int k=z_i-d; k<=(z_i+d);k++){
						if( (i>=0 && i<spatial_dim) && (j>=0 && j<spatial_dim) && (k>=0 && k<spatial_dim) ){
							int idx = k + j*spatial_dim + i*spatial_dim*spatial_dim;							
							float r2 = (x - i*res)*(x - i*res)+\
							(y - j*res)*(y - j*res)+\
							(z - k*res)*(z - k*res);
							type_volume[idx]+=exp(-r2/2.0);
						}
					}
				}
			}
		}
	}

__global__ void projectFromTensor(	double* coords, double* grad, uint* num_atoms_of_type, uint* offsets, float *volume,
                                    int spatial_dim, float res)
    /*
    Input:
        coords: coordinates in a flat array:
            coords: {protein1, ... proteinN}
            protein1: {atom_type1 .. atom_typeM}
            atom_type: {x1,y1,z1 .. xL,yL,zL}
        num_atoms_of_type: number of atoms in each atom_type 
        offsets: offset for coordinates for each atom_type volume
        volume: gradient to be projected on atoms
    Output: 
        grad: for each atom to store the gradient projection
    */
    {
		int d = 2;
		size_t func_index = threadIdx.x + blockIdx.x*blockDim.x;
		float *type_volume = volume + func_index * spatial_dim*spatial_dim*spatial_dim;
		double *atoms_coords = coords + offsets[func_index];
		uint n_atoms = num_atoms_of_type[func_index];
		for(int atom_idx = 0; atom_idx<n_atoms; atom_idx+=3){
			float 	x = atoms_coords[atom_idx],
					y = atoms_coords[atom_idx + 1],
					z = atoms_coords[atom_idx + 2];
            atoms_coords[atom_idx] = 0.0;
            atoms_coords[atom_idx+1] = 0.0;
            atoms_coords[atom_idx+2] = 0.0;
			int x_i = floor(x/res);
			int y_i = floor(y/res);
			int z_i = floor(z/res);
			for(int i=x_i-d; i<=(x_i+d);i++){
				for(int j=y_i-d; j<=(y_i+d);j++){
					for(int k=z_i-d; k<=(z_i+d);k++){
						if( (i>=0 && i<spatial_dim) && (j>=0 && j<spatial_dim) && (k>=0 && k<spatial_dim) ){
							int idx = k + j*spatial_dim + i*spatial_dim*spatial_dim;							
							float r2 = (x - i*res)*(x - i*res)+\
							(y - j*res)*(y - j*res)+\
							(z - k*res)*(z - k*res);
							atoms_coords[atom_idx] -= i*res*type_volume[idx]*exp(-r2/2.0);
                            atoms_coords[atom_idx + 1] -= j*res*type_volume[idx]*exp(-r2/2.0);
                            atoms_coords[atom_idx + 2] -= k*res*type_volume[idx]*exp(-r2/2.0);
						}
					}
				}
			}
		}
	}

void gpu_computeCoords2Volume(	double *coords,
                                uint *num_atoms_of_type,
							    uint *offsets, 
								float *volume,
								uint spatial_dim,
                                uint num_atom_types,
								float res){

	projectToTensor<<<1, num_atom_types>>>(	coords, num_atoms_of_type, offsets,
											volume, spatial_dim, res);

}

void gpu_computeVolume2Coords(	double *coords,
								double* grad,
                                uint *num_atoms_of_type,
							    uint *offsets, 
								float *volume,
								uint spatial_dim,
                                uint num_atom_types,
								float res){

	projectFromTensor<<<1, num_atom_types>>>(	coords, grad, num_atoms_of_type, offsets,
												volume, spatial_dim, res);

}