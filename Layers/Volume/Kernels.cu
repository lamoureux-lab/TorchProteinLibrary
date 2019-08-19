#include <Kernels.h>

template <typename T>
__global__ void projectToTensor(T* coords, int* num_atoms_of_type, int* offsets, T *volume, 
                                int spatial_dim, float res){
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
	int type_index = threadIdx.x;
	T *type_volume = volume + type_index * spatial_dim*spatial_dim*spatial_dim;
	T *atoms_coords = coords + 3*offsets[type_index];
	int n_atoms = num_atoms_of_type[type_index];
	for(int atom_idx = 0; atom_idx<3*n_atoms; atom_idx+=3){
		T 	x = atoms_coords[atom_idx],
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
						T r2 = (x - i*res)*(x - i*res)+\
						(y - j*res)*(y - j*res)+\
						(z - k*res)*(z - k*res);
						type_volume[idx]+=exp(-r2/2.0);
					}
				}
			}
		}
	}
}
template <typename T>
__global__ void projectFromTensor(T* coords, T* grad, int* num_atoms_of_type, int* offsets, T *volume,
                                  int spatial_dim, float res){
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
	int d = 2;
	int type_index = threadIdx.x;
	T *type_volume = volume + type_index * spatial_dim*spatial_dim*spatial_dim;
	T *atoms_coords = coords + 3*offsets[type_index];
	T *grad_coords = grad + 3*offsets[type_index];
	int n_atoms = num_atoms_of_type[type_index];
	for(int atom_idx = 0; atom_idx<3*n_atoms; atom_idx+=3){
		T 	x = atoms_coords[atom_idx],
			y = atoms_coords[atom_idx + 1],
			z = atoms_coords[atom_idx + 2];
		// grad_coords[atom_idx] = 0.0;
		// grad_coords[atom_idx+1] = 0.0;
		// grad_coords[atom_idx+2] = 0.0;
		int x_i = floor(x/res);
		int y_i = floor(y/res);
		int z_i = floor(z/res);
		
		for(int i=x_i-d; i<=(x_i+d);i++){
			for(int j=y_i-d; j<=(y_i+d);j++){
				for(int k=z_i-d; k<=(z_i+d);k++){
					if( (i>=0 && i<spatial_dim) && (j>=0 && j<spatial_dim) && (k>=0 && k<spatial_dim) ){
						int idx = k + j*spatial_dim + i*spatial_dim*spatial_dim;
						T r2 = (x - i*res)*(x - i*res)+\
						(y - j*res)*(y - j*res)+\
						(z - k*res)*(z - k*res);
						grad_coords[atom_idx] -= (x - i*res)*type_volume[idx]*exp(-r2/2.0);
						grad_coords[atom_idx + 1] -= (y-j*res)*type_volume[idx]*exp(-r2/2.0);
						grad_coords[atom_idx + 2] -= (z-k*res)*type_volume[idx]*exp(-r2/2.0);
					}
				}
			}
		}
		
	}
}


__global__ void selectFromTensor(float *features, 
								float* volume, int spatial_dim, 
								float *coords, int num_atoms, int max_num_atoms,
								float res){
/*
Input:
	volume: 3d array from which we select
	coords: coordinates in a flat array
	num_atoms: number of atoms 
	spatial_dim: volume 3d array real size
	res: volume 3d array resolution
		
Output: 
	features: for each atom to store the elements from the array
*/
	int feature_index = threadIdx.x;
	float *feature_volume = volume + feature_index * spatial_dim*spatial_dim*spatial_dim;
	float *feature_output = features + feature_index * max_num_atoms;
	for(int atom_idx = 0; atom_idx<num_atoms; atom_idx++){
		float 	x = floor(coords[3*atom_idx]/res),
				y = floor(coords[3*atom_idx + 1]/res),
				z = floor(coords[3*atom_idx + 2]/res);
		if( (x<spatial_dim && x>=0)&&(y<spatial_dim && y>=0)&&(z<spatial_dim && z>=0)){
			uint idx = z + y*spatial_dim + x*spatial_dim*spatial_dim;
			feature_output[atom_idx] = feature_volume[idx];
		}
	}
}

template <typename T>
void gpu_computeCoords2Volume(	T *coords,
                                int *num_atoms_of_type,
							    int *offsets, 
								T *volume,
								int spatial_dim,
                                int num_atom_types,
								float res){

	projectToTensor<T><<<1, num_atom_types>>>(	coords, num_atoms_of_type, offsets,
											volume, spatial_dim, res);

}
template <typename T>
void gpu_computeVolume2Coords(	T *coords,
								T* grad,
                                int *num_atoms_of_type,
							    int *offsets, 
								T *volume,
								int spatial_dim,
                                int num_atom_types,
								float res){

	projectFromTensor<T><<<1, num_atom_types>>>(coords, grad, num_atoms_of_type, offsets,
												volume, spatial_dim, res);

}

void gpu_selectFromTensor(	float *features, int num_features, 
							float* volume, int spatial_dim, 
							float *coords, int num_atoms, int max_num_atoms, 
							float res){

	selectFromTensor<<<1, num_features>>>(	features, 
											volume, spatial_dim, 
											coords, num_atoms, max_num_atoms,
											res);
}

template void gpu_computeVolume2Coords<float>(	float*, float*, int*, int*, float*, int, int, float);
template void gpu_computeVolume2Coords<double>(	double*, double*, int*, int*, double*, int, int, float);

template void gpu_computeCoords2Volume<float>(float*, int*, int*, float*, int, int, float);
template void gpu_computeCoords2Volume<double>(double*, int*, int*, double*, int, int, float);