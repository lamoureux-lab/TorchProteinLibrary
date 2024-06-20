#include <Kernels.h>

/*
template <typename T>
__global__ void projectToCell(T* coords, int num_atoms, T *volume, int spatial_dim, float res){
	uint d = 2;
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;
	
    if( !((i>=0 && i<spatial_dim) && (j>=0 && j<spatial_dim) && (k>=0 && k<spatial_dim)) )
        return;
    
    uint idx = k + j*spatial_dim + i*spatial_dim*spatial_dim;
    T result = 0.0;
    for(int atom_idx = 0; atom_idx<num_atoms; atom_idx++){
		T x = coords[3*atom_idx], y = coords[3*atom_idx + 1], z = coords[3*atom_idx + 2];
		
        int x_i = floor(x/res);
		int y_i = floor(y/res);
		int z_i = floor(z/res);

        if(__sad(x_i, i, 0)>d) continue;
        if(__sad(y_i, j, 0)>d) continue;
        if(__sad(z_i, k, 0)>d) continue;

        T r2 = (x - i*res)*(x - i*res)+(y - j*res)*(y - j*res)+(z - k*res)*(z - k*res);

		result += exp(-r2/2.0);
	}
    volume[idx] = result;
}

template <typename T>
void gpu_computeCoords2Volume(	T *coords,
                                int num_atoms,
								T *volume,
								int spatial_dim,
								float res){
	
	dim3 threadsPerBlock(4, 4, 4);
    dim3 numBlocks( spatial_dim/threadsPerBlock.x + 1,
                    spatial_dim/threadsPerBlock.y + 1,
                    spatial_dim/threadsPerBlock.z + 1);
	
	projectToCell<T><<<numBlocks,threadsPerBlock>>>(coords, num_atoms, volume, spatial_dim, res);
	
}


template <typename T>
__global__ void projectFromTensor(	T* coords, T* grad, int num_atoms, 
									T *volume, int spatial_dim, float res){
	uint d = 2;
	uint atom_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(atom_idx>num_atoms)return;
	
	T x = coords[3*atom_idx], y = coords[3*atom_idx + 1], z = coords[3*atom_idx + 2];
	int x_i = floor(x/res);
	int y_i = floor(y/res);
	int z_i = floor(z/res);
	
	T grad_x=0, grad_y=0, grad_z=0;
	for(int i=x_i-d; i<=(x_i+d);i++){
		for(int j=y_i-d; j<=(y_i+d);j++){
			for(int k=z_i-d; k<=(z_i+d);k++){
				if( (i>=0 && i<spatial_dim) && (j>=0 && j<spatial_dim) && (k>=0 && k<spatial_dim) ){
					int cell_idx = k + j*spatial_dim + i*spatial_dim*spatial_dim;
					T vol_value = volume[cell_idx];

					T r2 = (x - i*res)*(x - i*res)+(y - j*res)*(y - j*res)+(z - k*res)*(z - k*res);
					
					grad_x -= (x - i*res)*vol_value*exp(-r2/2.0);
					grad_y -= (y - j*res)*vol_value*exp(-r2/2.0);
					grad_z -= (z - k*res)*vol_value*exp(-r2/2.0);
				}
			}
		}
	}
	grad[3*atom_idx] = grad_x;
	grad[3*atom_idx + 1] = grad_y;
	grad[3*atom_idx + 2] = grad_z;
}

template <typename T>
void gpu_computeVolume2Coords(	T *coords,
								T* grad,
                                int num_atoms, 
								T *volume,
								int spatial_dim,
								float res){
	dim3 threadsPerBlock(64);
    dim3 numBlocks( num_atoms/threadsPerBlock.x + 1);

	projectFromTensor<T><<<numBlocks,threadsPerBlock>>>(coords, grad, num_atoms, volume, spatial_dim, res);
}
*/

__global__ void coordSelect(float *features, 
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

__global__ void coordSelectGrad(float *gradOutput, 
								float* gradInput, int spatial_dim, 
								float *coords, int num_atoms, int max_num_atoms,
								float res){
/*
Input:
	gradOutput: gradient of selected features
	coords: coordinates in a flat array
	num_atoms: number of atoms 
	spatial_dim: volume 3d array real size
	res: volume 3d array resolution
		
Output: 
	gradInput: 3d array from which we selected
*/
	int feature_index = threadIdx.x;
	float *feature_volume = gradInput + feature_index * spatial_dim*spatial_dim*spatial_dim;
	float *feature_grad = gradOutput + feature_index * max_num_atoms;
	for(int atom_idx = 0; atom_idx<num_atoms; atom_idx++){
		float 	x = floor(coords[3*atom_idx]/res),
				y = floor(coords[3*atom_idx + 1]/res),
				z = floor(coords[3*atom_idx + 2]/res);
		if( (x<spatial_dim && x>=0)&&(y<spatial_dim && y>=0)&&(z<spatial_dim && z>=0)){
			uint idx = z + y*spatial_dim + x*spatial_dim*spatial_dim;
			feature_volume[idx] = feature_grad[atom_idx];
		}
	}
}


void gpu_coordSelect(	float *features, int num_features, 
						float* volume, int spatial_dim, 
						float *coords, int num_atoms, int max_num_atoms, 
						float res){

	coordSelect<<<1, num_features>>>(	features, 
										volume, spatial_dim, 
										coords, num_atoms, max_num_atoms,
										res);
}

void gpu_coordSelectGrad(	float *gradOutput, int num_features, 
							float* gradInput, int spatial_dim, 
							float *coords, int num_atoms, int max_num_atoms, 
							float res){

	coordSelectGrad<<<1, num_features>>>(	gradOutput, 
											gradInput, spatial_dim, 
											coords, num_atoms, max_num_atoms,
											res);
}
/*
template void gpu_computeVolume2Coords<float>(	float*, float*, int, float*, int, float);
template void gpu_computeVolume2Coords<double>(	double*, double*, int, double*, int, float);

template void gpu_computeCoords2Volume<float>(float*, int, float*, int, float);
template void gpu_computeCoords2Volume<double>(double*, int, double*, int, float);
*/