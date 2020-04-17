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

__device__ 
int3 computeGridPos(float3 p, float res){
	int3 gridPos;
	gridPos.x = floor(p.x/res);
	gridPos.y = floor(p.y/res);
	gridPos.z = floor(p.z/res);
	return gridPos;
}

__device__ 
uint computeGridHash(int3 gridPos, int spatial_dim){
	return gridPos.z + gridPos.y * spatial_dim + gridPos.x * spatial_dim * spatial_dim;
}

__global__ 
void computeHash( 	uint *gridParticleHash, uint *gridParticleIndex, float *coords, 
					int num_atoms, int spatial_dim, float res, int d){
	uint atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint num_neighbours = (2*d+1)*(2*d+1)*(2*d+1);
	if (atom_idx>=num_atoms) return;
	
	int3 gridPos = computeGridPos(	make_float3(
										coords[3*atom_idx+0],
										coords[3*atom_idx+1],
										coords[3*atom_idx+2]), res);
	uint neighbour_idx = 0;
	for(int i=gridPos.x-d; i<=gridPos.x+d; i++){
		for(int j=gridPos.y-d; j<=gridPos.y+d; j++){
			for(int k=gridPos.z-d; k<=gridPos.z+d; k++){
				if( ((i>=0 && i<spatial_dim) && (j>=0 && j<spatial_dim) && (k>=0 && k<spatial_dim)) ){
					uint hash = computeGridHash(make_int3(i,j,k), spatial_dim);
					gridParticleHash[num_neighbours*atom_idx + neighbour_idx] = hash;
					gridParticleIndex[num_neighbours*atom_idx + neighbour_idx] = atom_idx;
					
				}
				neighbour_idx+=1;
				
			}
		}
	}
}


__global__ 
void reorderData(uint *cellStart, uint *cellEnd,
				float3 *sortedPos, 
				uint *gridParticleHash, uint *gridParticleIndex,
				float *coords, int num_atoms){

	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ uint sharedHash[]; 
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint hash;

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
			uint sortedIndex = gridParticleIndex[index];
			sortedPos[index] = make_float3(coords[3*sortedIndex+0], coords[3*sortedIndex+1], coords[3*sortedIndex+2]);
		}
	}
}


__global__ 
void evalCell(	float3 *sortedPos, float *volume, 
				uint *cellStart, uint *cellEnd, 
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
	
	float3 cell = make_float3(i*res, j*res, k*res);
    
    float result = 0.0;
    for(int atom_idx = start; atom_idx<end; atom_idx++){
		float3 atom = sortedPos[atom_idx];
        float r2 = (cell.x - atom.x)*(cell.x - atom.x)+(cell.y - atom.y)*(cell.y - atom.y)+(cell.z - atom.z)*(cell.z - atom.z);
		result += exp(-r2/2.0);
	}
    
    volume[cellHash] = result;
}

__global__ void printArrays(uint *a, uint *b, float *c, uint num_elem){
	for(int i=0; i<num_elem; i++){
		printf("%d: %d->%d (%.2f, %.2f, %.2f)\n", i, a[i], b[i], c[3*i+0], c[3*i+1], c[3*i+2]);
	}
}
__global__ void printArrays(uint *a, uint *b, uint num_elem){
	for(int i=0; i<num_elem; i++){
		printf("%d: %d->%d\n", i, a[i], b[i]);
	}
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

void gpu_BuildHashTable(float *coords,
						int num_atoms,
						float *volume,
						int spatial_dim,
						float res, 
						int d){
	// printf("Start\n");
	uint num_neighbours = (2*d+1)*(2*d+1)*(2*d+1);
	uint *gridParticleHash, *gridParticleIndex, *cellStart, *cellEnd;
	uint grid_size = spatial_dim*spatial_dim*spatial_dim;
	cudaMalloc(&gridParticleHash, num_neighbours*num_atoms*sizeof(uint));
	cudaMalloc(&gridParticleIndex, num_neighbours*num_atoms*sizeof(uint));
	cudaMalloc(&cellStart, grid_size*sizeof(uint));
	cudaMalloc(&cellEnd, grid_size*sizeof(uint));

	cudaMemset(gridParticleHash, 0xffffffff, num_neighbours*num_atoms*sizeof(uint));
	cudaMemset(gridParticleIndex, 0xffffffff, num_neighbours*num_atoms*sizeof(uint));
	cudaMemset(cellStart, 0xffffffff, grid_size*sizeof(uint));
	cudaMemset(cellEnd, 0xffffffff, grid_size*sizeof(uint));

	float3 *sortedPos;
	cudaMalloc(&sortedPos, num_neighbours*num_atoms*sizeof(float3));
	cudaMemset(sortedPos, 0, num_neighbours*num_atoms*sizeof(float3));
	
	uint numThreads, numBlocks;
	computeGridSize(num_atoms, 256, numBlocks, numThreads);
	// printf("Blocks %d threads %d\n", numBlocks, numThreads);
	
	computeHash<<<numBlocks, numThreads>>>(	gridParticleHash, gridParticleIndex,
											coords, num_atoms, spatial_dim, res, d);
	gpuErrchk( cudaPeekAtLastError() );
	// getLastCudaError("Kernel execution failed: compute hash");
	// printf("___________Hashes:\n");
	// printArrays<<<1,1>>>(gridParticleHash, gridParticleIndex, num_neighbours*num_atoms);
	
	
	thrust::sort_by_key(thrust::device_ptr<uint>(gridParticleHash),
                        thrust::device_ptr<uint>(gridParticleHash + num_neighbours*num_atoms),
                        thrust::device_ptr<uint>(gridParticleIndex));
	gpuErrchk( cudaPeekAtLastError() );
	// printf("___________Ordered Hashes:\n");
	// printArrays<<<1,1>>>(gridParticleHash, gridParticleIndex, num_neighbours*num_atoms);

	computeGridSize(num_neighbours*num_atoms, 256, numBlocks, numThreads);
	uint smemSize = sizeof(uint)*(numThreads+1);
	// printf("Blocks %d threads %d\n", numBlocks, numThreads);
	reorderData<<<numBlocks, numThreads, smemSize>>>(cellStart, cellEnd,
													sortedPos,
													gridParticleHash, gridParticleIndex,
													coords,	num_neighbours*num_atoms);
	gpuErrchk( cudaPeekAtLastError() );
	// printf("___________Ordered Data:\n");
	// printArrays<<<1,1>>>(gridParticleHash, gridParticleIndex, (float*)sortedPos, num_neighbours*num_atoms);
	// printArrays<<<1,1>>>(cellStart, cellEnd, grid_size);

	dim3 d3ThreadsPerBlock(4, 4, 4);
    dim3 d3NumBlocks( 	spatial_dim/d3ThreadsPerBlock.x + 1,
                    	spatial_dim/d3ThreadsPerBlock.y + 1,
                    	spatial_dim/d3ThreadsPerBlock.z + 1);
	evalCell<<<d3NumBlocks,d3ThreadsPerBlock>>>(sortedPos, volume, cellStart, cellEnd, spatial_dim, res);

	
}