#include <TH/TH.h>
#include <THC/THC.h>
#include "cPDBLoader.h"
#include <iostream>
#include <string>
#include <Kernels.h>

extern THCState *state;

extern bool int2bool(int var);
extern cVector3 getRandomTranslation(THGenerator *gen, double bound);

extern "C" {
    void PDB2VolumeLocal(   THByteTensor *filenames, THCudaTensor *volume, 
                            THCudaTensor *ca_coords, THCudaIntTensor *num_atoms, 
                            int rotate, int translate){
        THGenerator *gen = THGenerator_new();
 		THRandom_seed(gen);
        bool rot = int2bool(rotate);
        bool tran = int2bool(translate);
        
        if(filenames->nDimension == 2){
            int max_num_ca = 0;
            int batch_size = filenames->size[0];
            std::vector<cPDBLoader*> pdbs;
            pdbs.resize(batch_size);

            #pragma omp parallel for
            for(int i=0; i<batch_size; i++){
                
                THByteTensor *single_filename = THByteTensor_new();
                THCudaTensor *single_volume = THCudaTensor_new(state);
                
                THByteTensor_select(single_filename, filenames, 0, i);
                THCudaTensor_select(state, single_volume, volume, 0, i);
                
                std::string filename((const char*)THByteTensor_data(single_filename));
                pdbs[i] = new cPDBLoader(filename);
                pdbs[i]->computeBoundingBox();
                cVector3 center_box = (pdbs[i]->b0 + pdbs[i]->b1)*0.5;
                pdbs[i]->translate( -center_box );
                if(rot)
                    pdbs[i]->randRot(gen);
                cVector3 center_volume(single_volume->size[1]/2.0, single_volume->size[2]/2.0, single_volume->size[3]/2.0);
                pdbs[i]->translate(center_volume);
                if(tran)
                    pdbs[i]->randTrans(gen, single_volume->size[1]);
            
                int total_size = 3*pdbs[i]->getNumAtoms();
                int num_atom_types = 11;
                double coords[total_size];
                int num_atoms_of_type[num_atom_types], offsets[num_atom_types];
                
                pdbs[i]->reorder(coords, num_atoms_of_type, offsets);
                
                double *d_coords;
                int *d_num_atoms_of_type;
                int *d_offsets;
                cudaMalloc( (void**) &d_coords, total_size*sizeof(double) );
                cudaMalloc( (void**) &d_num_atoms_of_type, num_atom_types*sizeof(int) );
                cudaMalloc( (void**) &d_offsets, num_atom_types*sizeof(int) );

                cudaMemcpy( d_offsets, offsets, num_atom_types*sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy( d_num_atoms_of_type, num_atoms_of_type, num_atom_types*sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy( d_coords, coords, total_size*sizeof(double), cudaMemcpyHostToDevice);
                
                
                gpu_computeCoords2Volume(d_coords, d_num_atoms_of_type, d_offsets, THCudaTensor_data(state, single_volume), 
                                        single_volume->size[1], num_atom_types, 1.0);
                
                THByteTensor_free(single_filename);
                THCudaTensor_free(state, single_volume);
                
                cudaFree(d_coords);
		        cudaFree(d_num_atoms_of_type);
		        cudaFree(d_offsets);

                //Copy coordinates of C-alpha atoms
                THCudaIntTensor_set1d(state, num_atoms, i, pdbs[i]->res_nums.back());
                
                #pragma omp critical // Executed by one thread at a time
                {
                    if( max_num_ca < pdbs[i]->res_nums.back()){
                        max_num_ca = pdbs[i]->res_nums.back();
                    }
                }
                                
            }   
            // Resizing coordinates tensor 
            THCudaTensor_resize2d(state, ca_coords, batch_size, max_num_ca*3);
            
            #pragma omp parallel for
            for(int i=0; i<batch_size; i++){
                THCudaTensor *single_ca_coords = THCudaTensor_new(state);
                THCudaTensor_select(state, single_ca_coords, ca_coords, 0, i);

                std::string CA("CA");
                for(int j=0; j<pdbs[i]->r.size(); j++){
                    if(pdbs[i]->atom_names[j] == CA){
                        int res_id = pdbs[i]->res_nums[j] - 1;
                        THCudaTensor_set1d(state, single_ca_coords, res_id*3, pdbs[i]->r[j].v[0]);
                        THCudaTensor_set1d(state, single_ca_coords, res_id*3+1, pdbs[i]->r[j].v[1]);
                        THCudaTensor_set1d(state, single_ca_coords, res_id*3+2, pdbs[i]->r[j].v[2]);
                    }
                }

                THCudaTensor_free(state, single_ca_coords);
            }
            
            for (int i=0; i<batch_size; i++){
                delete pdbs[i];
            } 
            pdbs.clear();
        
        }else{
            std::cout<<"Not implemented"<<std::endl;
            throw("Not implemented");
        }
        THGenerator_free(gen);
    }
}