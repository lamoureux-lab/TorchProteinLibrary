#include "cConformation.h"
#include <iostream>
#include <string>
#include "nUtil.h"
#include "coordsTransform_interface.h"



void CoordsTranslate_forward(   torch::Tensor input_coords, 
                                torch::Tensor output_coords,
                                torch::Tensor T,
                                torch::Tensor num_atoms
                                ){
    CHECK_CPU_INPUT(input_coords);
    CHECK_CPU_INPUT(output_coords);
    CHECK_CPU_INPUT(T);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(input_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    int batch_size = input_coords.size(0);
    auto num_at = num_atoms.accessor<int,1>();
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_input_coords = input_coords[i];
        torch::Tensor single_output_coords = output_coords[i];
        auto aT = T.accessor<double,2>();
        cVector3<double> translation(aT[i][0], aT[i][1], aT[i][2]);
        translate(single_input_coords, translation, single_output_coords, num_at[i]);
    }
}
void CoordsRotate_forward(  torch::Tensor input_coords, 
                            torch::Tensor output_coords,
                            torch::Tensor R,
                            torch::Tensor num_atoms
                            ){
    CHECK_CPU_INPUT(input_coords);
    CHECK_CPU_INPUT(output_coords);
    CHECK_CPU_INPUT(R);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(input_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    
    int batch_size = input_coords.size(0);
    auto num_at = num_atoms.accessor<int,1>();
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_input_coords = input_coords[i];
        torch::Tensor single_output_coords = output_coords[i];
        torch::Tensor single_R = R[i];
        
        cMatrix33<double> _R = tensor2Matrix33<double>(single_R);
        rotate(single_input_coords, _R, single_output_coords, num_at[i]);
    }
}
void CoordsRotate_backward( torch::Tensor grad_output_coords, 
                            torch::Tensor grad_input_coords,
                            torch::Tensor R,
                            torch::Tensor num_atoms){
    CHECK_CPU_INPUT(grad_output_coords);
    CHECK_CPU_INPUT(grad_input_coords);
    CHECK_CPU_INPUT(R);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(grad_output_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
        
    int batch_size = grad_output_coords.size(0);
    auto num_at = num_atoms.accessor<int,1>();
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_grad_output_coords = grad_output_coords[i];
        torch::Tensor single_grad_input_coords = grad_input_coords[i];
        torch::Tensor single_R = R[i];
        
        cMatrix33<double> _R = tensor2Matrix33<double>(single_R);
        _R = _R.getTranspose();
        rotate(single_grad_output_coords, _R, single_grad_input_coords, num_at[i]);
    }
}
void getBBox(   torch::Tensor input_coords,
                torch::Tensor a, torch::Tensor b,
                torch::Tensor num_atoms){
    CHECK_CPU_INPUT(input_coords);
    CHECK_CPU_INPUT(a);
    CHECK_CPU_INPUT(b);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(input_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    
    int batch_size = input_coords.size(0);
    auto num_at = num_atoms.accessor<int,1>();
    #pragma omp parallel for num_threads(10)
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_input_coords = input_coords[i];
        torch::Tensor single_a = a[i];
        torch::Tensor single_b = b[i];
        
        cVector3<double> va(single_a.data<double>());
        cVector3<double> vb(single_b.data<double>());
        computeBoundingBox(single_input_coords, num_at[i], va, vb);
    }
}
void getRandomRotation( torch::Tensor R ){
    CHECK_CPU_INPUT(R);
    if(R.ndimension() != 3){
        ERROR("Incorrect input ndim");
    }

    int batch_size = R.size(0);
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_R = R[i];
        cMatrix33<double> rnd_R = getRandomRotation<double>();
        matrix2Tensor(rnd_R, single_R);                
    }
}
void getRotation( torch::Tensor R, torch::Tensor u ){
    CHECK_CPU_INPUT(R);
    CHECK_CPU_INPUT(u);
    if(R.ndimension() != 3 || u.ndimension() !=2 ){
        ERROR("Incorrect input ndim");
    }

    int batch_size = R.size(0);
    auto param = u.accessor<double,2>();

    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_R = R[i];
        cMatrix33<double> R = getRotation(param[i][0], param[i][1], param[i][2]);
        matrix2Tensor(R, single_R);
    }
}
void getRandomTranslation( torch::Tensor T, torch::Tensor a, torch::Tensor b, float volume_size){
    CHECK_CPU_INPUT(T);
    CHECK_CPU_INPUT(a);
    CHECK_CPU_INPUT(b);
    if(T.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    
    int batch_size = T.size(0);
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_T = T[i];
        torch::Tensor single_a = a[i];
        torch::Tensor single_b = b[i];
                
        cVector3<double> _a(single_a.data<double>());
        cVector3<double> _b(single_b.data<double>());
        cVector3<double> _T(single_T.data<double>());
        
        _T = getRandomTranslation(volume_size, _a, _b);
    }
}
