#include <TH/TH.h>
#include <THC/THC.h>
#include "cPDBLoader.h"
#include <iostream>
#include <string>
#include <Kernels.h>

extern THCState *state;

void project(double *coords, uint *num_atoms_of_type, uint *offsets, float *volume, uint spatial_dim){
    float res = 1.0;
    int d = 2;
    for(int atom_type = 0; atom_type<11; atom_type++){
        uint offset = offsets[atom_type];
        float *type_volume = volume + spatial_dim*spatial_dim*spatial_dim*atom_type;
        for(int atom_idx = 0; atom_idx<num_atoms_of_type[atom_type]; atom_idx++){
            float 	x = coords[3*(atom_idx + offset) + 0],
                    y = coords[3*(atom_idx + offset) + 1],
                    z = coords[3*(atom_idx + offset) + 2];
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
            }}}
        }
    }
}
cVector3 getRandomTranslation(THGenerator *gen, double bound){
    float u1 = THRandom_uniform(gen,-bound,bound);
    float u2 = THRandom_uniform(gen,-bound,bound);
    float u3 = THRandom_uniform(gen,-bound,bound);
    return cVector3(u1, u2, u3);
}
extern "C" {
    void PDB2Volume( THByteTensor *filenames, THFloatTensor *volume){
        THGenerator *gen = THGenerator_new();
 		THRandom_seed(gen);
        if(filenames->nDimension == 1){
            std::string filename((const char*)THByteTensor_data(filenames));
            cPDBLoader pdb(filename);
            cVector3 center_mass = pdb.getCenterMass() * (-1.0);
            pdb.translate(center_mass);
            pdb.randRot(gen);
            cVector3 center_volume(volume->size[1]/2.0, volume->size[2]/2.0, volume->size[3]/2.0);
            pdb.translate(center_volume);
            pdb.translate(getRandomTranslation(gen, volume->size[1]/4.0));

            double coords[3*pdb.getNumAtoms()];
            uint num_atoms_of_type[11], offsets[11];
            pdb.reorder(coords, num_atoms_of_type, offsets);

            project(coords, num_atoms_of_type, offsets, THFloatTensor_data(volume), volume->size[1]);

        }else if(filenames->nDimension == 2){
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<filenames->size[0]; i++){
                THByteTensor *single_filename = THByteTensor_new();
                THFloatTensor *single_volume = THFloatTensor_new();
                THByteTensor_select(single_filename, filenames, 0, i);
                THFloatTensor_select(single_volume, volume, 0, i);
                std::string filename((const char*)THByteTensor_data(single_filename));

                cPDBLoader pdb(filename);
                cVector3 center_mass = pdb.getCenterMass() * (-1.0);
                pdb.translate(center_mass);
                pdb.randRot(gen);
                cVector3 center_volume(single_volume->size[1]/2.0, single_volume->size[2]/2.0, single_volume->size[3]/2.0);
                pdb.translate(center_volume);
                pdb.translate(getRandomTranslation(gen, single_volume->size[1]/4.0));

                double coords[3*pdb.getNumAtoms()];
                uint num_atoms_of_type[11], offsets[11];
                pdb.reorder(coords, num_atoms_of_type, offsets);
                project(coords, num_atoms_of_type, offsets, THFloatTensor_data(single_volume), single_volume->size[1]);

                THByteTensor_free(single_filename);
                THFloatTensor_free(single_volume);
            }
            
        }
        THGenerator_free(gen);
    }


    void PDB2VolumeCUDA( THByteTensor *filenames, THCudaTensor *volume){
        THGenerator *gen = THGenerator_new();
 		THRandom_seed(gen);
        try{
        if(filenames->nDimension == 1){
            std::string filename((const char*)THByteTensor_data(filenames));
            cPDBLoader pdb(filename);
            cVector3 center_mass = pdb.getCenterMass() * (-1.0);
            pdb.translate(center_mass);
            pdb.randRot(gen);
            cVector3 center_volume(volume->size[1]/2.0, volume->size[2]/2.0, volume->size[3]/2.0);
            pdb.translate(center_volume);
            pdb.translate(getRandomTranslation(gen, volume->size[1]/4.0));

            uint total_size = 3*pdb.getNumAtoms();
            uint num_atom_types = 11;
            double coords[total_size];
            uint num_atoms_of_type[num_atom_types], offsets[num_atom_types];
            pdb.reorder(coords, num_atoms_of_type, offsets);

            double *d_coords;
            uint *d_num_atoms_of_type;
            uint *d_offsets;
            cudaMalloc( (void**) &d_coords, total_size*sizeof(double) );
            cudaMalloc( (void**) &d_num_atoms_of_type, num_atom_types*sizeof(uint) );
            cudaMalloc( (void**) &d_offsets, num_atom_types*sizeof(uint) );

            cudaMemcpy( d_offsets, offsets, num_atom_types*sizeof(uint), cudaMemcpyHostToDevice);
	        cudaMemcpy( d_num_atoms_of_type, num_atoms_of_type, num_atom_types*sizeof(uint), cudaMemcpyHostToDevice);
	        cudaMemcpy( d_coords, coords, total_size*sizeof(double), cudaMemcpyHostToDevice);

            gpu_computeCoords2Volume(d_coords, d_num_atoms_of_type, d_offsets, THCudaTensor_data(state, volume), volume->size[1], num_atom_types, 1.0);

            cudaFree(d_coords);
		    cudaFree(d_num_atoms_of_type);
		    cudaFree(d_offsets);

        }else if(filenames->nDimension == 2){
            #pragma omp parallel for num_threads(10)
            for(int i=0; i<filenames->size[0]; i++){
               
                THByteTensor *single_filename = THByteTensor_new();
                THCudaTensor *single_volume = THCudaTensor_new(state);
                THByteTensor_select(single_filename, filenames, 0, i);
                THCudaTensor_select(state, single_volume, volume, 0, i);
                std::string filename((const char*)THByteTensor_data(single_filename));
                
                cPDBLoader pdb(filename);
                cVector3 center_mass = pdb.getCenterMass() * (-1.0);
                pdb.translate(center_mass);
                pdb.randRot(gen);
                cVector3 center_volume(single_volume->size[1]/2.0, single_volume->size[2]/2.0, single_volume->size[3]/2.0);
                pdb.translate(center_volume);
                pdb.translate(getRandomTranslation(gen, single_volume->size[1]/4.0));
            
                uint total_size = 3*pdb.getNumAtoms();
                uint num_atom_types = 11;
                double coords[total_size];
                uint num_atoms_of_type[num_atom_types], offsets[num_atom_types];
                
                pdb.reorder(coords, num_atoms_of_type, offsets);
                
                double *d_coords;
                uint *d_num_atoms_of_type;
                uint *d_offsets;
                cudaMalloc( (void**) &d_coords, total_size*sizeof(double) );
                cudaMalloc( (void**) &d_num_atoms_of_type, num_atom_types*sizeof(uint) );
                cudaMalloc( (void**) &d_offsets, num_atom_types*sizeof(uint) );

                cudaMemcpy( d_offsets, offsets, num_atom_types*sizeof(uint), cudaMemcpyHostToDevice);
                cudaMemcpy( d_num_atoms_of_type, num_atoms_of_type, num_atom_types*sizeof(uint), cudaMemcpyHostToDevice);
                cudaMemcpy( d_coords, coords, total_size*sizeof(double), cudaMemcpyHostToDevice);
                

                gpu_computeCoords2Volume(d_coords, d_num_atoms_of_type, d_offsets, THCudaTensor_data(state, single_volume), 
                                        single_volume->size[1], num_atom_types, 1.0);
                
                THByteTensor_free(single_filename);
                THCudaTensor_free(state, single_volume);
                cudaFree(d_coords);
		        cudaFree(d_num_atoms_of_type);
		        cudaFree(d_offsets);
                

            }
            
        }
        }catch(const std::exception& e){
            std::cout<<e.what()<<std::endl;
            exit(1);
        }
        THGenerator_free(gen);
    }
}