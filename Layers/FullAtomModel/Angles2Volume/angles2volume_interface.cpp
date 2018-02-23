#include <TH/TH.h>
#include <THC/THC.h>
#include "cPDBLoader.h"
#include "cConformation.h"
#include <iostream>
#include <string>
#include "nUtil.h"
#include <Kernels.h>

bool int2bool(int add_terminal){
    bool add_term;
    if(add_terminal == 1){
        add_term = true;
    }else if(add_terminal == 0){
        add_term = false;
    }else{
        std::cout<<"unknown add_terminal = "<<add_terminal<<std::endl;
        throw std::string("unknown add_terminal");
    }
    return add_term;
}

extern THCState *state;

void copy2TH(cMatrix33 src, THDoubleTensor *dst){
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            THDoubleTensor_set2d(dst, i, j, src.m[i][j]);
}
void copy2TH(cVector3 src, THDoubleTensor *dst){
    for(int i=0; i<3; i++)
        THDoubleTensor_set1d(dst, i, src.v[i]);
}
cMatrix33 copy4TH(THDoubleTensor *src){
    cMatrix33 dst;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            dst.m[i][j] = THDoubleTensor_get2d(src, i, j);
    return dst;
}
cVector3 copy4TH(THDoubleTensor *src){
    cVector3 dst;
    for(int i=0; i<3; i++)
        dst.v[i] = THDoubleTensor_get1d(src, i);
    return dst;
}
extern "C" {
    
    void Angles2Volume_forward( const char* sequence,
                                THDoubleTensor *input_angles, 
                                THCudaTensor *output_volume,
                                int add_terminal,
                                THDoubleTensor *R_ext,
                                THDoubleTensor *T_ext){
        THGenerator *gen = THGenerator_new();
 		THRandom_seed(gen);
        
        if(input_angles->nDimension == 2){
            
            std::string aa(sequence);
            uint length = aa.length();
            bool add_term = int2bool(add_terminal);
            uint num_atoms = ProtUtil::getNumAtoms(aa, add_term);
            
            THDoubleTensor *dummy_grad = THDoubleTensor_newWithSize2d(input_angles->size[0], input_angles->size[1]);
            THDoubleTensor *output_coords = THDoubleTensor_newWithSize1d(3*num_atoms);
            
            cConformation conf( aa, THDoubleTensor_data(input_angles), THDoubleTensor_data(dummy_grad),
                                length, THDoubleTensor_data(output_coords));

            THDoubleTensor_free(dummy_grad);
            
            cVector3 b0, b1;
            ProtUtil::computeBoundingBox(output_coords, b0, b1);
            cVector3 center_box = (b0 + b1)*0.5;
            ProtUtil::translate( output_coords, -center_box);
            cMatrix33 R = ProtUtil::getRandomRotation(gen);
            copy2TH(R, R_ext);
            ProtUtil::rotate(output_coords, R);
            cVector3 center_volume(output_volume->size[1]/2.0, output_volume->size[2]/2.0, output_volume->size[3]/2.0);
            ProtUtil::translate(output_coords, center_volume);
            cVector3 T = ProtUtil::getRandomTranslation(gen, output_volume->size[1], b0, b1);
            copy2TH(T, T_ext);
            ProtUtil::translate(output_coords, T);

            uint total_size = 3*num_atoms;
            uint num_atom_types = 11;
            double coords[total_size];
            uint num_atoms_of_type[num_atom_types], offsets[num_atom_types];
            std::vector<uint> atom_types;
            atom_types.resize(num_atoms);

            uint num_atoms_added[num_atom_types];
            for(int i=0;i<num_atom_types;i++){
                num_atoms_of_type[i] = 0;
                num_atoms_added[i] = 0;
                offsets[i] = 0;
            }
            
            for(uint i=0; i<conf.groups.size(); i++){
                for(uint j=0; j<conf.groups[i]->atomNames.size(); j++){
                    std::string res_name = ProtUtil::convertRes1to3(conf.groups[i]->residueName);
                    uint idx = conf.groups[i]->atomIndexes[j];
                    uint type = ProtUtil::get11AtomType( res_name, conf.groups[i]->atomNames[j], false);
                    atom_types[idx] = type;
                    num_atoms_of_type[type]+=1;
            }}
            for(uint i=1;i<num_atom_types;i++){
                offsets[i] = offsets[i-1] + num_atoms_of_type[i-1];
            }

            for(uint i=0; i<conf.groups.size(); i++){
                for(uint j=0; j<conf.groups[i]->atomNames.size(); j++){
                    uint idx = conf.groups[i]->atomIndexes[j];
                    uint type = atom_types[idx];
                    cVector3 r_dst(coords + 3*(offsets[type] + num_atoms_added[type]));
                    cVector3 r_src(THDoubleTensor_data(output_coords) + 3*idx);
                    r_dst = r_src[i];
                    num_atoms_added[type]+=1;
                }
            }    

            double *d_coords;
            uint *d_num_atoms_of_type;
            uint *d_offsets;
            cudaMalloc( (void**) &d_coords, total_size*sizeof(double) );
            cudaMalloc( (void**) &d_num_atoms_of_type, num_atom_types*sizeof(uint) );
            cudaMalloc( (void**) &d_offsets, num_atom_types*sizeof(uint) );

            cudaMemcpy( d_offsets, offsets, num_atom_types*sizeof(uint), cudaMemcpyHostToDevice);
	        cudaMemcpy( d_num_atoms_of_type, num_atoms_of_type, num_atom_types*sizeof(uint), cudaMemcpyHostToDevice);
	        cudaMemcpy( d_coords, coords, total_size*sizeof(double), cudaMemcpyHostToDevice);

            gpu_computeCoords2Volume(d_coords, d_num_atoms_of_type, d_offsets, THCudaTensor_data(state, output_volume), output_volume->size[1], num_atom_types, 1.0);

            cudaFree(d_coords);
		    cudaFree(d_num_atoms_of_type);
		    cudaFree(d_offsets);
            THDoubleTensor_free(output_coords);
        }
        THGenerator_free(gen);
    }



    void Angles2Volume_backward( const char* sequence,
                                THDoubleTensor *input_angles, 
                                THCudaTensor *grad_volume,
                                THDoubleTensor *grad_angles,
                                int add_terminal,
                                THDoubleTensor *R_ext,
                                THDoubleTensor *T_ext){
        THGenerator *gen = THGenerator_new();
 		THRandom_seed(gen);
        
        if(input_angles->nDimension == 2){
            
            std::string aa(sequence);
            uint length = aa.length();
            bool add_term = int2bool(add_terminal);
            uint num_atoms = ProtUtil::getNumAtoms(aa, add_term);
            
            
            THDoubleTensor *output_coords = THDoubleTensor_newWithSize1d(3*num_atoms);
            cConformation conf( aa, THDoubleTensor_data(input_angles), THDoubleTensor_data(grad_angles),
                                length, THDoubleTensor_data(output_coords));

            cVector3 b0, b1;
            ProtUtil::computeBoundingBox(output_coords, b0, b1);
            cVector3 center_box = (b0 + b1)*0.5;
            ProtUtil::translate( output_coords, -center_box);
            cMatrix33 R = copy4TH(R_ext);      
            ProtUtil::rotate(output_coords, R);
            cVector3 center_volume(grad_volume->size[1]/2.0, grad_volume->size[2]/2.0, grad_volume->size[3]/2.0);
            ProtUtil::translate(output_coords, center_volume);
            cVector3 T = copy4TH(T_ext);
            ProtUtil::translate(output_coords, T);

            uint total_size = 3*num_atoms;
            uint num_atom_types = 11;
            double coords[total_size], grad_coords[total_size];
            uint num_atoms_of_type[num_atom_types], offsets[num_atom_types];
            std::vector<uint> atom_types;
            atom_types.resize(num_atoms);

            uint num_atoms_added[num_atom_types];
            for(int i=0;i<num_atom_types;i++){
                num_atoms_of_type[i] = 0;
                num_atoms_added[i] = 0;
                offsets[i] = 0;
            }
            
            for(uint i=0; i<conf.groups.size(); i++){
                for(uint j=0; j<conf.groups[i]->atomNames.size(); j++){
                    std::string res_name = ProtUtil::convertRes1to3(conf.groups[i]->residueName);
                    uint idx = conf.groups[i]->atomIndexes[j];
                    uint type = ProtUtil::get11AtomType( res_name, conf.groups[i]->atomNames[j], false);
                    atom_types[idx] = type;
                    num_atoms_of_type[type]+=1;
            }}
            for(uint i=1;i<num_atom_types;i++){
                offsets[i] = offsets[i-1] + num_atoms_of_type[i-1];
            }

            for(uint i=0; i<conf.groups.size(); i++){
                for(uint j=0; j<conf.groups[i]->atomNames.size(); j++){
                    uint idx = conf.groups[i]->atomIndexes[j];
                    uint type = atom_types[idx];
                    cVector3 r_dst(coords + 3*(offsets[type] + num_atoms_added[type]));
                    cVector3 r_src(THDoubleTensor_data(output_coords) + 3*idx);
                    r_dst = r_src[i];
                    num_atoms_added[type]+=1;
                }
            }    

            double *d_coords, *d_grad;
            uint *d_num_atoms_of_type;
            uint *d_offsets;
            cudaMalloc( (void**) &d_coords, total_size*sizeof(double) );
            cudaMalloc( (void**) &d_grad, total_size*sizeof(double) );
            cudaMalloc( (void**) &d_num_atoms_of_type, num_atom_types*sizeof(uint) );
            cudaMalloc( (void**) &d_offsets, num_atom_types*sizeof(uint) );

            cudaMemcpy( d_offsets, offsets, num_atom_types*sizeof(uint), cudaMemcpyHostToDevice);
	        cudaMemcpy( d_num_atoms_of_type, num_atoms_of_type, num_atom_types*sizeof(uint), cudaMemcpyHostToDevice);
	        cudaMemcpy( d_coords, coords, total_size*sizeof(double), cudaMemcpyHostToDevice);

            gpu_computeVolume2Coords(d_coords, d_grad, d_num_atoms_of_type, d_offsets, THCudaTensor_data(state, grad_volume), grad_volume->size[1], num_atom_types, 1.0);

            cudaMemcpy( grad_coords, d_grad, total_size*sizeof(double), cudaMemcpyHostToDevice);
            
            
            conf.backward(conf.root, THDoubleTensor_data(grad_atoms));

            cudaFree(d_coords);
            cudaFree(d_grad);
		    cudaFree(d_num_atoms_of_type);
		    cudaFree(d_offsets);
            THDoubleTensor_free(output_coords);
        }
        THGenerator_free(gen);
    }
}