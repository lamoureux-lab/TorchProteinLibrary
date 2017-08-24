#include "cCoords2RMSD.h"
#include <iostream>
#include <math.h>
#include "../RMSDKernels.h"
#include <stdlib.h>


cCoords2RMSD::cCoords2RMSD( THCState *state, 
                            THCudaTensor *re_coordinates_src, 
                            THCudaTensor *re_coordinates_dst,
                            THCudaTensor *U_coordinates_src,
                            THCudaTensor *Ut_coordinates_dst,
                            THCudaTensor *rot_mat_t, int length
                        ){
    this->state = state;
    this->re_coordinates_src = re_coordinates_src;
    this->re_coordinates_dst = re_coordinates_dst;
    this->U_coordinates_src = U_coordinates_src;
    this->Ut_coordinates_dst = Ut_coordinates_dst;

    
    this->rot_mat_t = rot_mat_t;
    this->angles_length = length;
}

cCoords2RMSD::~cCoords2RMSD(){
    
}

float cCoords2RMSD::computeForward(THCudaTensor *coordinates_src, THCudaTensor *coordinates_dst){
    THCudaDoubleTensor *T = THCudaDoubleTensor_newWithSize2d(state, 4, 4);
    centroid_src = THCudaTensor_newWithSize1d(state, 3);
    centroid_dst = THCudaTensor_newWithSize1d(state, 3);
    THDoubleTensor *vec = THDoubleTensor_newWithSize2d(4,4);
    THDoubleTensor *eig = THDoubleTensor_newWithSize1d(4);
    rot_mat = THCudaTensor_newWithSize2d(state, 3,3);
    
    //computing centroids
    cpu_computeCentroid(THCudaTensor_data(state, coordinates_src), THCudaTensor_data(state, centroid_src), this->angles_length);
    cpu_computeCentroid(THCudaTensor_data(state, coordinates_dst), THCudaTensor_data(state, centroid_dst), this->angles_length);
    //centering the proteins
    cpu_centerCoords(THCudaTensor_data(state, coordinates_src), THCudaTensor_data(state, centroid_src),
                     THCudaTensor_data(state, re_coordinates_src), this->angles_length);
    cpu_centerCoords(THCudaTensor_data(state, coordinates_dst), THCudaTensor_data(state, centroid_dst),
                     THCudaTensor_data(state, re_coordinates_dst), this->angles_length);
    //correlation matrix T
    cpu_correlationMatrix(  THCudaTensor_data(state, re_coordinates_src),
                            THCudaTensor_data(state, re_coordinates_dst),
                            THCudaDoubleTensor_data(state, T),
                            this->angles_length);
    
    //computing eigenvectors and eigenvalues on CPU: CPU is not implemented in Torch, need Magma!!
    THDoubleTensor *cpu_T = fromGPUDouble(state, T);
    THDoubleTensor_resize2d(cpu_T,4,4);
    // for(int i=0; i<4; i++){
    //     for(int j=0;j<4;j++){
    //         std::cout<<THFloatTensor_get2d(cpu_T, i, j)<<", ";
    //     }
    //     std::cout<<"\n";
    // }
    //soving eigenvalues problem
    THDoubleTensor_syev(eig, vec, cpu_T, "V", "U");
    //getting maximum eigenvalue and eigenvector
    float max_eig = THDoubleTensor_get1d(eig,0);
    int max_eig_ind = 0;
    float q[4];
    for(int i=0; i<4; i++){
        // std::cout<<"Eig "<<i<<" = "<<THFloatTensor_get1d(eig,i)<<"\t";
        // if(fabs(THFloatTensor_get1d(eig,i))>=fabs(max_eig)){
        if(THDoubleTensor_get1d(eig,i)>=max_eig){
            max_eig = THDoubleTensor_get1d(eig,i);
            max_eig_ind = i;
            // std::cout<<"Eigvector:\n";
            for(int j=0;j<4;j++){
                q[j] = THDoubleTensor_get2d(vec,j,i);
                // std::cout<<q[j]<<"\t";
            }
            // std::cout<<"\n";
        }
    }
    THDoubleTensor_free(cpu_T);
   
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

    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            Ut[3*i+j] = U[3*j+i];
        }}
    
    //computing R2 coefficient
    double R2 = 0.0;
    THCudaDoubleTensor *R2_tmp = THCudaDoubleTensor_newWithSize1d(state, 3);
    cpu_computeR2(THCudaTensor_data(state, re_coordinates_src), this->angles_length, THCudaDoubleTensor_data(state, R2_tmp));
    R2+=THCudaDoubleTensor_sumall(state, R2_tmp);
    cpu_computeR2(THCudaTensor_data(state, re_coordinates_dst), this->angles_length, THCudaDoubleTensor_data(state, R2_tmp));
    R2+=THCudaDoubleTensor_sumall(state, R2_tmp);
    THCudaDoubleTensor_free(state, R2_tmp);

    rmsd = (R2 - 2.0*fabs(max_eig))/(this->angles_length+1);
    toGPUTensor(state, U, rot_mat);
    toGPUTensor(state, Ut, rot_mat_t);
    if(rmsd<0){
    // if(1){
        std::cout<<"GPU rmsd = "<<rmsd<<"\n";
        std::cout<<"GPU R2 = "<<R2<<"\n";
        std::cout<<"GPU max_eig = "<<max_eig<<"\n";
        std::cout<<"U:\n";
        THFloatTensor *cpu_src = fromGPU(state, coordinates_src);
        THFloatTensor *cpu_dst = fromGPU(state, coordinates_dst);
        std::cout<<"SRC:"<<std::endl;
        for(int i=0; i<(this->angles_length+1);i++){
            std::cout<<THFloatTensor_get1d(cpu_src, 3*i)<<", "<<THFloatTensor_get1d(cpu_src, 3*i+1)<<", "<<THFloatTensor_get1d(cpu_src, 3*i+2)<<",\n";
        }
        std::cout<<"DST:"<<std::endl;
        for(int i=0; i<(this->angles_length+1);i++){
            std::cout<<THFloatTensor_get1d(cpu_dst, 3*i)<<", "<<THFloatTensor_get1d(cpu_dst, 3*i+1)<<", "<<THFloatTensor_get1d(cpu_dst, 3*i+2)<<",\n";
        }
        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                std::cout<<U[3*i+j]<<"\t";
            }
            std::cout<<"\n";
        }
        std::cout<<"Angles length = "<<this->angles_length<<"\n";
        std::cout<<"Eigvec\n";
        for(int i=0; i<4; i++){
            std::cout<<q[i]<<"\n";
        }
        std::cout<<"RotMatT:"<<rot_mat_t->nDimension<<std::endl;
        // std::cout<<"cRMSD: src"<<THCudaTensor_sumall(state,coordinates_src)<<std::endl;
        // std::cout<<"cRMSD: dst"<<THCudaTensor_sumall(state,coordinates_dst)<<std::endl;
        // std::cout<<"cRMSD: centroidsrc"<<THCudaTensor_sumall(state,centroid_src)<<std::endl;
        // std::cout<<"cRMSD: centroiddst"<<THCudaTensor_sumall(state,centroid_dst)<<std::endl;
        // std::cout<<"cRMSD: resrc"<<THCudaTensor_sumall(state,re_coordinates_src)<<std::endl;
        // std::cout<<"cRMSD: redst"<<THCudaTensor_sumall(state,re_coordinates_dst)<<std::endl;
        exit(-1);
    }
    
    
    cpu_transformCoordinates(   THCudaTensor_data(state, re_coordinates_dst), 
                                THCudaTensor_data(state, Ut_coordinates_dst),
                                THCudaTensor_data(state, rot_mat_t),
                                this->angles_length );
    
    cpu_transformCoordinates(   THCudaTensor_data(state, re_coordinates_src), 
                                THCudaTensor_data(state, U_coordinates_src),
                                THCudaTensor_data(state, rot_mat),
                                this->angles_length );
    THCudaDoubleTensor_free(state, T);
    THCudaTensor_free(state, centroid_src);
    THCudaTensor_free(state, centroid_dst);
    THDoubleTensor_free(vec);
    THDoubleTensor_free(eig);
    THCudaTensor_free(state, rot_mat);
    return rmsd;
}

void cCoords2RMSD::computeBackward(THCudaTensor *gradInput, float gradOutput){
    // std::cout<<this->angles_length<<std::endl;
    // cpu_transformCoordinates(   THCudaTensor_data(state, re_coordinates_dst), 
    //                             THCudaTensor_data(state, Ut_coordinates_dst),
    //                             THCudaTensor_data(state, rot_mat_t),
    //                             this->angles_length );
    // THCudaTensor_cadd(state, gradInput, re_coordinates_src, -1, Ut_coordinates_dst);
    
    float mult = 1.0/(this->angles_length+1);
    cpu_mulAndSub(  THCudaTensor_data(state, gradInput), 
                    THCudaTensor_data(state, re_coordinates_src), 
                    THCudaTensor_data(state, Ut_coordinates_dst), 
                    mult,
                    this->angles_length);
    // THCudaTensor_mul(state, gradInput, gradInput, gradOutput * mult );
    // std::cout<<THCudaTensor_sumall(state,gradInput)<<std::endl;
}
