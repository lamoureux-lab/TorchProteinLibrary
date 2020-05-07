#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "HashKernel.h"

//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}

template <typename T>
__device__ int3 computeGridPos(T p_x, T p_y, T p_z, float res){
	int3 gridPos;
	gridPos.x = floor(p_x/res);
	gridPos.y = floor(p_y/res);
	gridPos.z = floor(p_z/res);
	return gridPos;
}

__device__ long computeGridHash(int3 gridPos, int spatial_dim){
	return gridPos.z + gridPos.y * spatial_dim + gridPos.x * spatial_dim * spatial_dim;
}

template <typename T>
__global__ void computeHash(long *gridParticleHash, long *gridParticleIndex, T *coords, 
					        int num_atoms, int spatial_dim, float res, int d){
	uint atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint num_neighbours = (2*d+1)*(2*d+1)*(2*d+1);
	if (atom_idx>=num_atoms) return;
    
	int3 gridPos = computeGridPos<T>(coords[3*atom_idx+0], coords[3*atom_idx+1], coords[3*atom_idx+2], res);
	uint neighbour_idx = 0;
	for(int i=gridPos.x-d; i<=gridPos.x+d; i++){
		for(int j=gridPos.y-d; j<=gridPos.y+d; j++){
			for(int k=gridPos.z-d; k<=gridPos.z+d; k++){
				if( ((i>=0 && i<spatial_dim) && (j>=0 && j<spatial_dim) && (k>=0 && k<spatial_dim)) ){
					long hash = computeGridHash(make_int3(i,j,k), spatial_dim);
					gridParticleHash[num_neighbours*atom_idx + neighbour_idx] = hash;
					gridParticleIndex[num_neighbours*atom_idx + neighbour_idx] = atom_idx;
				}
				neighbour_idx+=1;
			}
		}
	}
}

template <typename T>
__global__ void reorderData(long *cellStart, long *cellEnd,
                            T *sortedPos, 
                            long *gridParticleHash, long *gridParticleIndex,
                            T *coords, int num_atoms){

	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ long sharedHash[]; 
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	long hash;

	if(index < num_atoms){
		hash = gridParticleHash[index];
		sharedHash[threadIdx.x + 1] = hash;
		if(threadIdx.x == 0 && index > 0){
			sharedHash[0] = gridParticleHash[index-1];
		}
		
	}

	cg::sync(cta);

	if(index < num_atoms){
		
		if(sharedHash[threadIdx.x] == 0xffffffff){
			return;
		}
		
		if(index == 0 || hash != sharedHash[threadIdx.x]){
			if(hash != 0xffffffff)
				cellStart[hash] = index;
			if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
		}
		if(index == num_atoms - 1 && hash != 0xffffffff){
			cellEnd[hash] = index + 1;
		}
		if(hash != 0xffffffff){
			long sortedIndex = gridParticleIndex[index];
			sortedPos[3*index+0] = coords[3*sortedIndex+0];
            sortedPos[3*index+1] = coords[3*sortedIndex+1];
            sortedPos[3*index+2] = coords[3*sortedIndex+2];
		}
	}
}


template <typename T>
__global__ void evalCell(T *sortedPos, T *volume, 
                        long *cellStart, long *cellEnd, 
                        int spatial_dim, float res){

    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if( !((i>=0 && i<spatial_dim) && (j>=0 && j<spatial_dim) && (k>=0 && k<spatial_dim)) )
        return;
	
	uint cellHash = k + j*spatial_dim + i*spatial_dim*spatial_dim;
	uint start = cellStart[cellHash];
	if(start == 0xffffffff)
		return;
	uint end = cellEnd[cellHash];
	
	T cell_x = i*res;
    T cell_y = j*res;
    T cell_z = k*res;
    
    T result = 0.0, r2;
    T atom_x, atom_y, atom_z;
    for(int atom_idx = start; atom_idx<end; atom_idx++){
		atom_x = sortedPos[3*atom_idx+0];
        atom_y = sortedPos[3*atom_idx+1];
        atom_z = sortedPos[3*atom_idx+2];
        r2 = (cell_x - atom_x)*(cell_x - atom_x)+(cell_y - atom_y)*(cell_y - atom_y)+(cell_z - atom_z)*(cell_z - atom_z);
		result += exp(-r2/2.0);
	}
    
    volume[cellHash] = result;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T>
void gpu_computeCoords2Volume(	T *coords,
                                int num_atoms,
								T *volume,
								int spatial_dim,
								float res,
                                int d,
								long *gridParticleHash,
								long *gridParticleIndex,
								long *cellStart,
								long *cellStop,
								T *sortedPos){
    if(num_atoms == 0)return;                                
	uint num_neighbours = (2*d+1)*(2*d+1)*(2*d+1);
	// uint *gridParticleHash, *gridParticleIndex, *cellStart, *cellEnd;
    // T *sortedPos;
	// uint grid_size = spatial_dim*spatial_dim*spatial_dim;

	// cudaMalloc(&sortedPos, num_neighbours*num_atoms*3*sizeof(T));
    // cudaMalloc(&gridParticleHash, num_neighbours*num_atoms*sizeof(uint));
	// cudaMalloc(&gridParticleIndex, num_neighbours*num_atoms*sizeof(uint));
	// cudaMalloc(&cellStart, grid_size*sizeof(uint));
	// cudaMalloc(&cellEnd, grid_size*sizeof(uint));
        
	// cudaMemset(gridParticleHash, 0xffffffff, num_neighbours*num_atoms*sizeof(uint));
	// cudaMemset(gridParticleIndex, 0xffffffff, num_neighbours*num_atoms*sizeof(uint));
	// cudaMemset(cellStart, 0xffffffff, grid_size*sizeof(uint));
	// cudaMemset(cellEnd, 0xffffffff, grid_size*sizeof(uint));
	// cudaMemset(sortedPos, 0.0, num_neighbours*num_atoms*3*sizeof(T));
	
	uint numThreads, numBlocks;
	computeGridSize(num_atoms, 64, numBlocks, numThreads);	
	computeHash<T><<<numBlocks, numThreads>>>(	gridParticleHash, gridParticleIndex,
											    coords, num_atoms, spatial_dim, res, d);
	gpuErrchk( cudaPeekAtLastError() );
		
	thrust::sort_by_key(thrust::device_ptr<long>(gridParticleHash),
                        thrust::device_ptr<long>(gridParticleHash + num_neighbours*num_atoms),
                        thrust::device_ptr<long>(gridParticleIndex));
	gpuErrchk( cudaPeekAtLastError() );

	computeGridSize(num_neighbours*num_atoms, 64, numBlocks, numThreads);
	uint smemSize = sizeof(long)*(numThreads+1);
	reorderData<T><<<numBlocks, numThreads, smemSize>>>(cellStart, cellStop,
                                                        sortedPos,
                                                        gridParticleHash, gridParticleIndex,
                                                        coords,	num_neighbours*num_atoms);
	gpuErrchk( cudaPeekAtLastError() );

	dim3 d3ThreadsPerBlock(4, 4, 4);
    dim3 d3NumBlocks( 	spatial_dim/d3ThreadsPerBlock.x + 1,
                    	spatial_dim/d3ThreadsPerBlock.y + 1,
                    	spatial_dim/d3ThreadsPerBlock.z + 1);
	evalCell<T><<<d3NumBlocks,d3ThreadsPerBlock>>>(sortedPos, volume, cellStart, cellStop, spatial_dim, res);

    // cudaFree(gridParticleHash);
    // cudaFree(gridParticleIndex);
    // cudaFree(cellStart);
    // cudaFree(cellEnd);
    // cudaFree(sortedPos);
}



template <typename T>
__global__ void projectFromTensor(	T* coords, T* grad, int num_atoms, 
									T *volume, int spatial_dim, float res, int d){
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
								int spatial_dim, float res, int d){
	dim3 threadsPerBlock(64);
    dim3 numBlocks( num_atoms/threadsPerBlock.x + 1);

	projectFromTensor<T><<<numBlocks,threadsPerBlock>>>(coords, grad, num_atoms, volume, spatial_dim, res, d);
}


template void gpu_computeVolume2Coords<float>(	float*, float*, int, float*, int, float, int);
template void gpu_computeVolume2Coords<double>(	double*, double*, int, double*, int, float, int);

template void gpu_computeCoords2Volume<float>(float*, int, float*, int, float, int, long*, long*, long*, long*, float*);
template void gpu_computeCoords2Volume<double>(double*, int, double*, int, float, int, long*, long*, long*, long*, double*);
