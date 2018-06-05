#include <THC/THC.h>
#include "../RMSDKernels.h"
#include <iostream>

extern THCState *state;
float R = 3.8;
extern "C" {
    void Coords2RMSD_forward(    THCudaDoubleTensor *re_coordinates_src, THCudaDoubleTensor *re_coordinates_dst, 
                                THCudaDoubleTensor *output, THCudaIntTensor *num_atoms,
                                THCudaDoubleTensor *Ut_coordinates_dst
                            ){
        if(re_coordinates_src->nDimension == 2){
            int batch_size = num_atoms->size[0];
            THCudaDoubleTensor *T = THCudaDoubleTensor_newWithSize3d(state, batch_size, 4, 4);
            THDoubleTensor *vec = THDoubleTensor_newWithSize3d(batch_size, 4,4);
            THDoubleTensor *eig = THDoubleTensor_newWithSize2d(batch_size, 4);
            THCudaDoubleTensor *rot_mat_t = THCudaDoubleTensor_newWithSize3d(state, batch_size, 3, 3);
                                    
            //correlation matrix T
            cpu_correlationMatrix(  THCudaDoubleTensor_data(state, re_coordinates_src),
                                    THCudaDoubleTensor_data(state, re_coordinates_dst),
                                    THCudaDoubleTensor_data(state, T),
                                    THCudaIntTensor_data(state, num_atoms), batch_size, re_coordinates_src->size[1]);

            
            //computing eigenvectors and eigenvalues on CPU: CPU is not implemented in Torch, need Magma!!
            THDoubleTensor *cpu_T = fromGPU(state, T);
            THDoubleTensor_resize3d(cpu_T, batch_size, 4, 4);
            
            for(int i=0;i<batch_size;i++){
                THDoubleTensor *cpu_T_single = THDoubleTensor_new();
                THDoubleTensor *eig_single = THDoubleTensor_new();
                THDoubleTensor *vec_single = THDoubleTensor_new();
                THCudaDoubleTensor *rot_mat_t_single = THCudaDoubleTensor_new(state);
                THCudaDoubleTensor *re_coords_src_single = THCudaDoubleTensor_new(state);
                THCudaDoubleTensor *re_coords_dst_single = THCudaDoubleTensor_new(state);
                
                THDoubleTensor_select(cpu_T_single, cpu_T, 0, i);
                THDoubleTensor_select(eig_single, eig, 0, i);
                THDoubleTensor_select(vec_single, vec, 0, i);
                THCudaDoubleTensor_select(state, rot_mat_t_single, rot_mat_t, 0, i);
                THCudaDoubleTensor_select(state, re_coords_src_single, re_coordinates_src, 0, i);
                THCudaDoubleTensor_select(state, re_coords_dst_single, re_coordinates_dst, 0, i);

                //soving eigenvalues problem
                THDoubleTensor_syev(eig_single, vec_single, cpu_T_single, "V", "U");
                //getting maximum eigenvalue and eigenvector
                double max_eig = THDoubleTensor_get1d(eig_single,0);
                int max_eig_ind = 0;
                double q[4], U[16], Ut[16];
                for(int j=0; j<4; j++){
                    if(THDoubleTensor_get1d(eig_single,j)>=max_eig){
                        max_eig = THDoubleTensor_get1d(eig_single,j);
                        max_eig_ind = j;
                        for(int k=0;k<4;k++){
                            q[k] = THDoubleTensor_get2d(vec_single,k,j);
                        }
                    }
                }
                
                // rotation matrix
                U[0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
                U[1] = 2.0*(q[1]*q[2] - q[0]*q[3]);
                U[2] = 2.0*(q[1]*q[3] + q[0]*q[2]);

                U[3] = 2.0*(q[1]*q[2] + q[0]*q[3]);
                U[4] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
                U[5] = 2.0*(q[2]*q[3] - q[0]*q[1]);

                U[6] = 2.0*(q[1]*q[3] - q[0]*q[2]);
                U[7] = 2.0*(q[2]*q[3] + q[0]*q[1]);
                U[8] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];

                for(int j=0;j<3;j++){
                    for(int k=0;k<3;k++){
                        Ut[3*j+k] = U[3*k+j];
                }}
                
                //computing R2 coefficient
                int num_coords = THCudaIntTensor_get1d(state, num_atoms, i);
                double R2 = 0.0;
                THCudaDoubleTensor *R2_tmp = THCudaDoubleTensor_newWithSize1d(state, 3);
                cpu_computeR2(THCudaDoubleTensor_data(state, re_coords_src_single), num_coords, THCudaDoubleTensor_data(state, R2_tmp));
                R2+=THCudaDoubleTensor_sumall(state, R2_tmp);
                cpu_computeR2(THCudaDoubleTensor_data(state, re_coords_dst_single), num_coords, THCudaDoubleTensor_data(state, R2_tmp));
                R2+=THCudaDoubleTensor_sumall(state, R2_tmp);
                THCudaDoubleTensor_free(state, R2_tmp);
                
                double rmsd = (R2 - 2.0*fabs(max_eig))/(num_coords);

                THCudaDoubleTensor_set1d(state, output, i, rmsd);
                
                toGPUTensor(state, Ut, rot_mat_t_single);
                
                THDoubleTensor_free(cpu_T_single);
                THDoubleTensor_free(eig_single);
                THDoubleTensor_free(vec_single);
                THCudaDoubleTensor_free(state, rot_mat_t_single);
                THCudaDoubleTensor_free(state, re_coords_src_single);
                THCudaDoubleTensor_free(state, re_coords_dst_single);

            }

            cpu_transformCoordinates(   THCudaDoubleTensor_data(state, re_coordinates_dst),
                                        THCudaDoubleTensor_data(state, Ut_coordinates_dst),
                                        THCudaDoubleTensor_data(state, rot_mat_t),
                                        batch_size, re_coordinates_dst->size[1]);


            THDoubleTensor_free(cpu_T);
            THDoubleTensor_free(vec);
            THDoubleTensor_free(eig);
            THCudaDoubleTensor_free(state, rot_mat_t);
            
            
        }
    }
}