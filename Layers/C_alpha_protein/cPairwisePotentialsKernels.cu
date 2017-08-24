#include "cPairwisePotentialsKernels.h"
#include "cMathCUDAKernels.h"


__global__ void gpu_computePairwiseForcesPP(    float *d_paircoords,    //input: pairwise coordinates 3 x maxlen x maxlen
                                                int *input_types,       //input: atom types maxlen
                                                float *output_forces,   //output: forces maxlen x 3
                                                float *potentials,      //params: potentials num_types x num_types x num_bins
                                                int num_types,
                                                int num_bins,
                                                float resolution,
                                                int L,
                                                int Lmax){
    int atoms_size = L+1;
    int max_atoms = Lmax+1;
	uint i = threadIdx.x; // 0..atoms_size
	int plane_stride = max_atoms*max_atoms;
    int itype = input_types[i];
    output_forces[3*i] = 0.0;
    output_forces[3*i+1] = 0.0;
    output_forces[3*i+2] = 0.0;

	for(int j=0; j<atoms_size; j++){
        if(j==i) // no self-interaction, counting interactions once
            continue;
        int jtype = input_types[j];
        float rx_ij = d_paircoords[0*plane_stride + i*max_atoms + j];
        float ry_ij = d_paircoords[1*plane_stride + i*max_atoms + j];
        float rz_ij = d_paircoords[2*plane_stride + i*max_atoms + j];
        float mod_rij = sqrt(rx_ij*rx_ij + ry_ij*ry_ij + rz_ij*rz_ij);
        int bin_idx = int(mod_rij/resolution);
        if(bin_idx>num_bins) // outside potentials interaction = 0
            continue;
        int potential_idx = 0;
        if(jtype >= itype){ //taking only one triangle of the matrix
            potential_idx = (itype*num_types + jtype)*num_bins;
        }else{
            potential_idx = (jtype*num_types + itype)*num_bins;
        }
        atomicAdd(output_forces+3*i, potentials[potential_idx + bin_idx] * rx_ij / mod_rij);
        atomicAdd(output_forces+3*i+1, potentials[potential_idx + bin_idx] * ry_ij / mod_rij);
        atomicAdd(output_forces+3*i+2, potentials[potential_idx + bin_idx] * rz_ij / mod_rij);
    }

}
__global__ void gpu_computeGradientForcesPP(    float *forces_grads,    //input: gradients of forces 3 x maxlen
                                                float *d_dpaircoords,   //output: 3 x maxlen x maxlen
                                                float *d_paircoords,      //params: coordinates maxlen x 3
                                                float *potentials,      //params: potentials num_types x num_types x num_bins
                                                int *input_types,       //params: atom types maxlen

                                                int num_types,
                                                int num_bins,
                                                float resolution,
                                                int L,
                                                int Lmax){
    int atoms_size = L+1;
    int max_atoms = Lmax+1;
	uint i = blockIdx.x; // 0..atoms_size
    uint j = threadIdx.x; // 0..atoms_size
    if (i==j) //no self-interaction
        return;

	int plane_stride = max_atoms*max_atoms;

    float rx_ij = d_paircoords[0*plane_stride + i*max_atoms + j];
    float ry_ij = d_paircoords[1*plane_stride + i*max_atoms + j];
    float rz_ij = d_paircoords[2*plane_stride + i*max_atoms + j];

    int itype = input_types[i];
    int jtype = input_types[j];
    int potential_idx = 0;
    if(jtype >= itype){ //taking only one triangle of the matrix
        potential_idx = (itype*num_types + jtype)*num_bins;
    }else{
        potential_idx = (jtype*num_types + itype)*num_bins;
    }
    float mod_rij2 = rx_ij*rx_ij + ry_ij*ry_ij + rz_ij*rz_ij;
    float mod_rij = sqrt(mod_rij2);
    if(mod_rij < 0.1){
        mod_rij = 0.1;
    }
    int bin_idx = int(mod_rij/resolution);
    
    
    float dphi = 0.0;
    if( bin_idx >= num_bins){
        return;
    }else if( (bin_idx+1) >= num_bins){
        dphi = (potentials[potential_idx + bin_idx] - potentials[potential_idx + bin_idx - 1])/resolution;
    }else if( (bin_idx-1) >= 0){
        dphi = (potentials[potential_idx + bin_idx + 1] - potentials[potential_idx + bin_idx - 1])/(2.0*resolution);
    }else{
        dphi = (potentials[potential_idx + bin_idx + 1] - potentials[potential_idx + bin_idx])/resolution;
    }
    float phi = potentials[potential_idx + bin_idx];

    float trace = forces_grads[3*i]*rx_ij + forces_grads[3*i+1]*ry_ij + forces_grads[3*i+2]*rz_ij; 
    float dLdx = trace*( dphi - phi/mod_rij )*(rx_ij/mod_rij2) + forces_grads[3*i]*phi/mod_rij;
    float dLdy = trace*( dphi - phi/mod_rij )*(ry_ij/mod_rij2) + forces_grads[3*i+1]*phi/mod_rij;
    float dLdz = trace*( dphi - phi/mod_rij )*(rz_ij/mod_rij2) + forces_grads[3*i+2]*phi/mod_rij;

    atomicAdd(d_dpaircoords+0*plane_stride + i*max_atoms + j, dLdx);
    atomicAdd(d_dpaircoords+1*plane_stride + i*max_atoms + j, dLdy);
    atomicAdd(d_dpaircoords+2*plane_stride + i*max_atoms + j, dLdz);
}

__global__ void gpu_computeGradientPotentials(  float *forces_grads,    //input: gradients of forces 3 x maxlen
                                                float *d_dpotentials,   //output: num_types x num_types x num_bins
                                                float *d_paircoords,      //params: coordinates maxlen x 3
                                                int *input_types,       //params: atom types maxlen
                                                int num_types,
                                                int num_bins,
                                                float resolution,
                                                int L, int Lmax){
    int atoms_size = L+1;
    int max_atoms = Lmax+1;
	uint k = blockIdx.x; // 0..num_types
    uint l = threadIdx.x; // 0..num_types
    int plane_stride = max_atoms*max_atoms;
    for(int i=0; i<atoms_size; i++){
        for(int j=0; j<atoms_size; j++){
            if(i==j)continue;
            int itype = input_types[i];
            int jtype = input_types[j];
            if(itype!=k || jtype!=l)continue;
            
            int potential_idx = 0;
            if(jtype >= itype){ //taking only one triangle of the matrix
                potential_idx = (itype*num_types + jtype)*num_bins;
            }else{
                potential_idx = (jtype*num_types + itype)*num_bins;
            }
            float rx_ij = d_paircoords[0*plane_stride + i*max_atoms + j];
            float ry_ij = d_paircoords[1*plane_stride + i*max_atoms + j];
            float rz_ij = d_paircoords[2*plane_stride + i*max_atoms + j];
            float mod_rij2 = rx_ij*rx_ij + ry_ij*ry_ij + rz_ij*rz_ij;
            float mod_rij = sqrt(mod_rij2);
            if(mod_rij < 0.1){
                mod_rij = 0.1;
            }
            int bin_idx = int(mod_rij/resolution);
            if(bin_idx>=num_bins) // outside potentials interaction = 0
                continue;
            float dLdphi = (forces_grads[3*i]*rx_ij + forces_grads[3*i+1]*ry_ij + forces_grads[3*i+2]*rz_ij)/mod_rij;
            // printf("%d %d %d -> %f\n",itype, jtype, bin_idx, dLdphi);
            atomicAdd(d_dpotentials + potential_idx + bin_idx, dLdphi);
     }}

                        
}

// __global__ void gpu_computePairwiseDist(    float *d_paircoords,    //input: pairwise coordinates 3 x maxlen x maxlen
//                                             int *input_types,       //input: atom types maxlen
//                                             float *output_dist,   //output: forces maxlen x 3
//                                             int num_types,
//                                             int num_bins,
//                                             float resolution,
//                                             int L,
//                                             int Lmax){
//     int atoms_size = L+1;
//     int max_atoms = Lmax+1;
//     uint i = blockIdx.x; 
//     uint at2 = floorf(threadIdx.x/num_bins);    //atom_type2
//     uint bin_idx = threadIdx.x - at2*num_bins;    //number of bin
// 	int plane_stride = max_atoms*max_atoms;
        
// 	for(int j=0; j<atoms_size; j++){
//         if(i==j)
//             continue;
//         if(input_types[i]==at2){
//             float rx_ij = d_paircoords[0*plane_stride + i*max_atoms + j];
//             float ry_ij = d_paircoords[1*plane_stride + i*max_atoms + j];
//             float rz_ij = d_paircoords[2*plane_stride + i*max_atoms + j];
//             float mod_rij = sqrt(rx_ij*rx_ij + ry_ij*ry_ij + rz_ij*rz_ij);
//             if(mod_rij<0.0001)
//                 continue;
//             uint cbin_idx = int(mod_rij/resolution);
//             if(cbin_idx!=bin_idx)
//                 continue;
            
//             //total length max_atoms*num_types*num_bins*3;
//             uint global_ind = i*num_types*num_bins*3 + at2*num_bins*3 + bin_idx*3;

//             output_dist[global_ind] += rx_ij/mod_rij;
//             output_dist[global_ind+1] += ry_ij/mod_rij;
//             output_dist[global_ind+2] += rz_ij/mod_rij;
//         }
//     }

// }

__global__ void gpu_computePairwiseDist(    float *d_paircoords,    //input: pairwise coordinates 3 x maxlen x maxlen
                                            int *input_types,       //input: atom types maxlen
                                            float *output_dist,   //output: forces maxlen x 3
                                            int num_types,
                                            int num_bins,
                                            float resolution,
                                            int L,
                                            int Lmax){
    float sigma = 2.0;
    int atoms_size = L+1;
    int max_atoms = Lmax+1;
    uint i = blockIdx.x; 
    uint at2 = floorf(threadIdx.x);    //atom_type2
	int plane_stride = max_atoms*max_atoms;
        
	for(int j=0; j<atoms_size; j++){
        if(i==j)
            continue;
        if(input_types[j]==at2){
            float rx_ij = d_paircoords[0*plane_stride + i*max_atoms + j];
            float ry_ij = d_paircoords[1*plane_stride + i*max_atoms + j];
            float rz_ij = d_paircoords[2*plane_stride + i*max_atoms + j];
            float mod_rij = sqrt(rx_ij*rx_ij + ry_ij*ry_ij + rz_ij*rz_ij);
            for(int bin_idx=0; bin_idx<num_bins; bin_idx++){
                float r_k = bin_idx*resolution;
                                
                //total length max_atoms*num_types*num_bins;
                uint global_ind = i*num_types*num_bins + at2*num_bins + bin_idx;
                output_dist[global_ind] += exp( -(r_k - mod_rij)*(r_k - mod_rij)/sigma);
            }
        }
    }

}

// __global__ void gpu_backwardPairwiseDist(   float *gradInput_pairs,    //input: coordinates 3 x maxlen x maxlen
//                                             float *gradOutput_dist,       //input: atom types maxlen
//                                             int *input_types,   //output: forces maxlen x 3
//                                             float *input_pairs,
//                                             int num_types,
//                                             int num_bins,
//                                             float resolution,
//                                             int L, int Lmax){
//     int atoms_size = L+1;
//     int max_atoms = Lmax+1;
//     uint i = blockIdx.x; 
//     uint at2 = floorf(threadIdx.x/num_bins);    //atom_type2
//     uint bin_idx = threadIdx.x - at2*num_bins;    //number of bin
// 	int plane_stride = max_atoms*max_atoms;
        
// 	for(int j=0; j<atoms_size; j++){
//         if(i==j)
//             continue;
//         if(input_types[i]==at2){
//             float rx_ij = input_pairs[0*plane_stride + i*max_atoms + j];
//             float ry_ij = input_pairs[1*plane_stride + i*max_atoms + j];
//             float rz_ij = input_pairs[2*plane_stride + i*max_atoms + j];
//             float mod_rij = sqrt(rx_ij*rx_ij + ry_ij*ry_ij + rz_ij*rz_ij);
//             if(mod_rij<0.0001)
//                 continue;
//             uint cbin_idx = int(mod_rij/resolution);
//             if(cbin_idx!=bin_idx)
//                 continue;
//             //total length max_atoms*num_types*num_bins*3;
//             uint global_ind = i*num_types*num_bins*3 + at2*num_bins*3 + bin_idx*3;
//             float go_x = gradOutput_dist[global_ind];
//             float go_y = gradOutput_dist[global_ind+1];
//             float go_z = gradOutput_dist[global_ind+2];

//             float c1 = 1.0/(mod_rij);
//             float c2 = 1.0/(mod_rij*mod_rij*mod_rij);
            
//             float res_x = go_x*(c1 - rx_ij*rx_ij*c2) + go_y*(- rx_ij*ry_ij*c2) + go_z*(- rx_ij*rz_ij*c2);
//             float res_y = go_x*(- ry_ij*rx_ij*c2) + go_y*( c1 - ry_ij*ry_ij*c2) + go_z*(- ry_ij*rz_ij*c2);
//             float res_z = go_x*(- rz_ij*rx_ij*c2) + go_y*( - rz_ij*ry_ij*c2) + go_z*( c1 - rz_ij*rz_ij*c2);
            
//             atomicAdd(gradInput_pairs + 0*plane_stride + i*max_atoms + j, res_x);
//             atomicAdd(gradInput_pairs + 1*plane_stride + i*max_atoms + j, res_y);
//             atomicAdd(gradInput_pairs + 2*plane_stride + i*max_atoms + j, res_z);
//         }
//     }

// }

__global__ void gpu_backwardPairwiseDist(   float *gradInput_pairs,    //input: coordinates 3 x maxlen x maxlen
                                            float *gradOutput_dist,       //input: atom types maxlen
                                            int *input_types,   //output: forces maxlen x 3
                                            float *input_pairs,
                                            int num_types,
                                            int num_bins,
                                            float resolution,
                                            int L, int Lmax){
    float sigma = 2.0;
    int atoms_size = L+1;
    int max_atoms = Lmax+1;
    uint i = blockIdx.x; 
    uint at2 = floorf(threadIdx.x);    //atom_type2
	int plane_stride = max_atoms*max_atoms;
        
	for(int j=0; j<atoms_size; j++){
        if(i==j)
            continue;
        if(input_types[j]==at2){
            float rx_ij = input_pairs[0*plane_stride + i*max_atoms + j];
            float ry_ij = input_pairs[1*plane_stride + i*max_atoms + j];
            float rz_ij = input_pairs[2*plane_stride + i*max_atoms + j];
            float mod_rij = sqrt(rx_ij*rx_ij + ry_ij*ry_ij + rz_ij*rz_ij);
            for(int bin_idx=0; bin_idx<num_bins; bin_idx++){
                float r_k = bin_idx*resolution;
                                                            
                //total length max_atoms*num_types*num_bins;
                uint global_ind = i*num_types*num_bins + at2*num_bins + bin_idx;
                float go = gradOutput_dist[global_ind];
                float c1 = 1.0/(mod_rij+0.00001);
                float exp_coef = go*2*((r_k-mod_rij)/sigma)*exp( -(r_k - mod_rij)*(r_k - mod_rij)/sigma);
                atomicAdd(gradInput_pairs + 0*plane_stride + i*max_atoms + j, (rx_ij*c1)*exp_coef);
                atomicAdd(gradInput_pairs + 1*plane_stride + i*max_atoms + j, (ry_ij*c1)*exp_coef);
                atomicAdd(gradInput_pairs + 2*plane_stride + i*max_atoms + j, (rz_ij*c1)*exp_coef);
                
            }
        }
    }
}


void cpu_computePairwiseForcesPP(   float *d_paircoords,    //input: coordinates 3 x maxlen x maxlen
                                    int *input_types,       //input: atom types maxlen
                                    float *output_forces,   //output: forces maxlen x 3
                                    float *potentials,      //params: potentials num_types x num_types x num_bins
                                    int num_types,
                                    int num_bins,
                                    float resolution,
                                    int L, int Lmax){

    gpu_computePairwiseForcesPP <<< 1, L+1 >>>( d_paircoords, input_types, output_forces, potentials, num_types, num_bins, resolution, L, Lmax);
}

void cpu_computeGradientForcesPP(   float *forces_grads,    //input: gradients of forces 3 x maxlen
                                    float *d_dpaircoords,   //output: maxlen x 3
                                    float *d_paircoords,      //params: coordinates maxlen x 3
                                    float *potentials,      //params: potentials num_types x num_types x num_bins
                                    int *input_types,       //params: atom types maxlen
                                    int num_types,
                                    int num_bins,
                                    float resolution,
                                    int L, int Lmax){
    gpu_computeGradientForcesPP <<< L+1, L+1 >>>( forces_grads, d_dpaircoords, d_paircoords, potentials, input_types, num_types, num_bins, resolution, L, Lmax);
}

void cpu_computeGradientPotentials(     float *forces_grads,    //input: gradients of forces 3 x maxlen
                                        float *d_dpotentials,   //output: num_types x num_types x num_bins
                                        float *d_paircoords,      //params: coordinates maxlen x 3
                                        int *input_types,       //params: atom types maxlen
                                        int num_types,
                                        int num_bins,
                                        float resolution,
                                        int L, int Lmax){

    gpu_computeGradientPotentials <<< num_types, num_types >>>(forces_grads, d_dpotentials, d_paircoords, input_types, num_types, num_bins,resolution,L,Lmax);

}       


void cpu_computePairwiseDistributions(float *d_paircoords,    //input: coordinates 3 x maxlen x maxlen
                                    int *input_types,       //input: atom types maxlen
                                    float *output_dist,   //output: forces maxlen x 3
                                    int num_types,
                                    int num_bins,
                                    float resolution,
                                    int L, int Lmax){
    
    gpu_computePairwiseDist <<< L+1, num_types >>>(d_paircoords, input_types, output_dist, num_types, num_bins, resolution,L,Lmax);

}       

void cpu_backwardPairwiseDistributions( float *gradInput_pairs,    //input: coordinates 3 x maxlen x maxlen
                                        float *gradOutput_dist,       //input: atom types maxlen
                                        int *input_types,   //output: forces maxlen x 3
                                        float *input_pairs,
                                        int num_types,
                                        int num_bins,
                                        float resolution,
                                        int L, int Lmax){
    
    gpu_backwardPairwiseDist <<< L+1, num_bins*num_types >>>(gradInput_pairs, gradOutput_dist, input_types, input_pairs, num_types, num_bins, resolution,L,Lmax);

}       