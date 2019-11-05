#include "coords2elec_interface.h"
#include <iostream>
#include "nUtil.h"
#include "KernelsElectrostatics.h"

void Coords2Eps_forward(torch::Tensor coords, torch::Tensor assigned_params, torch::Tensor num_atoms, torch::Tensor eps, 
                        float resolution, float ion_size, float wat_size, float asigma, int d){
    CHECK_GPU_INPUT_TYPE(coords, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(assigned_params, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(eps, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    
    int batch_size = num_atoms.size(0);
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        std::cout<<"batch_idx="<<i<<std::endl;
        torch::Tensor single_coords = coords[i];
        torch::Tensor single_params = assigned_params[i];
        torch::Tensor single_eps = eps[i];
        gpu_computePartialSumFaces( single_coords.data<float>(), 
                                    single_params.data<float>(),
                                    num_atoms[i].item().toInt(),
                                    single_eps.data<float>(),
                                    single_eps.size(1), //box_size
                                    resolution, 
                                    ion_size, wat_size, asigma, d);
    }
    
}


void Coords2Q_forward(torch::Tensor coords, torch::Tensor assigned_params, torch::Tensor num_atoms, torch::Tensor Q, float resolution){
    CHECK_GPU_INPUT_TYPE(coords, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(assigned_params, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(Q, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);

    int batch_size = num_atoms.size(0);
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_coords = coords[i];
        torch::Tensor single_params = assigned_params[i];
        torch::Tensor single_Q = Q[i];
    
        gpu_computeSumCells(    single_coords.data<float>(), 
                                single_params.data<float>(),
                                num_atoms[i].item().toInt(),
                                single_Q.data<float>(),
                                single_Q.size(0), //box_size
                                resolution);
    }
}

void Coords2Eps_backward(   torch::Tensor gradOutput, torch::Tensor gradInput, 
                            torch::Tensor coords, torch::Tensor assigned_params, torch::Tensor num_atoms,
                            float resolution){
    CHECK_GPU_INPUT_TYPE(gradOutput, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(gradInput, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(coords, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(assigned_params, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(num_atoms, torch::kInt);
    
}

void QEps2Phi_forward(torch::Tensor Q, torch::Tensor Eps, torch::Tensor Phi, float resolution, float kappa02){
    CHECK_GPU_INPUT_TYPE(Q, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(Eps, torch::kFloat);
    CHECK_GPU_INPUT_TYPE(Phi, torch::kFloat);

    int batch_size = Q.size(0);
    // #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_Q = Q[i];
        torch::Tensor single_Eps = Eps[i];
        torch::Tensor single_Phi = Phi[i];
        gpu_computePhi( single_Q.data<float>(), 
                        single_Eps.data<float>(), 
                        single_Phi.data<float>(), 
                        single_Q.size(0), //box_size
                        resolution, kappa02);
    }
}

