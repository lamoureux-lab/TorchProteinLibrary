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
        AT_DISPATCH_FLOATING_TYPES(input_coords.type(), "CoordsTranslate_forward", ([&]{
            auto aT = T.accessor<scalar_t,2>();
            cVector3<scalar_t> translation(aT[i][0], aT[i][1], aT[i][2]);
            translate<scalar_t>(single_input_coords, translation, single_output_coords, num_at[i]);
        }));
    }
}

void CoordsTranslate_backward(  torch::Tensor grad_output_coords, 
                                torch::Tensor grad_input_coords,
                                torch::Tensor T,
                                torch::Tensor num_atoms
                                ){
    CHECK_CPU_INPUT(grad_output_coords);
    CHECK_CPU_INPUT(grad_input_coords);
    CHECK_CPU_INPUT(T);
    CHECK_CPU_INPUT_TYPE(num_atoms, torch::kInt);
    if(grad_output_coords.ndimension() != 2){
        ERROR("Incorrect input ndim");
    }
    int batch_size = grad_output_coords.size(0);
    auto num_at = num_atoms.accessor<int,1>();
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_grad_input_coords = grad_input_coords[i];
        torch::Tensor single_grad_output_coords = grad_output_coords[i];
        AT_DISPATCH_FLOATING_TYPES(single_grad_output_coords.type(), "CoordsTranslate_backward", ([&]{
            for(int j=0; j<num_at[i]; j++){
                cVector3<scalar_t> r_in(single_grad_input_coords.data<scalar_t>() + 3*j);
                cVector3<scalar_t> r_out(single_grad_output_coords.data<scalar_t>() + 3*j);
                r_in = r_out;
            }
        }));
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
        
        AT_DISPATCH_FLOATING_TYPES(input_coords.type(), "CoordsRotate_forward", ([&]{
            cMatrix33<scalar_t> _R = tensor2Matrix33<scalar_t>(single_R);
            rotate<scalar_t>(single_input_coords, _R, single_output_coords, num_at[i]);
        }));
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
        AT_DISPATCH_FLOATING_TYPES(grad_output_coords.type(), "CoordsRotate_backward", ([&]{
            cMatrix33<scalar_t> _R = tensor2Matrix33<scalar_t>(single_R);
            _R = _R.getTranspose();
            rotate<scalar_t>(single_grad_output_coords, _R, single_grad_input_coords, num_at[i]);
        }));
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
        
        AT_DISPATCH_FLOATING_TYPES(input_coords.type(), "getBBox", ([&]{
            cVector3<scalar_t> va(single_a.data<scalar_t>());
            cVector3<scalar_t> vb(single_b.data<scalar_t>());
            computeBoundingBox<scalar_t>(single_input_coords, num_at[i], va, vb);
        }));
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
        AT_DISPATCH_FLOATING_TYPES(R.type(), "getRandomRotation", ([&]{
            cMatrix33<scalar_t> rnd_R = getRandomRotation<scalar_t>();
            matrix2Tensor<scalar_t>(rnd_R, single_R);                
        }));
    }
}
void getRotation( torch::Tensor R, torch::Tensor u ){
    CHECK_CPU_INPUT(R);
    CHECK_CPU_INPUT(u);
    if(R.ndimension() != 3 || u.ndimension() !=2 ){
        ERROR("Incorrect input ndim");
    }

    int batch_size = R.size(0);
    
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        torch::Tensor single_R = R[i];
        AT_DISPATCH_FLOATING_TYPES(R.type(), "getRotation", ([&]{ 
            auto param = u.accessor<scalar_t,2>();
            cMatrix33<scalar_t> Rot = getRotation(param[i][0], param[i][1], param[i][2]);
            matrix2Tensor<scalar_t>(Rot, single_R);
        }));
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
        
        AT_DISPATCH_FLOATING_TYPES(T.type(), "getRandomTranslation", ([&]{        
            cVector3<scalar_t> _a(single_a.data<scalar_t>());
            cVector3<scalar_t> _b(single_b.data<scalar_t>());
            cVector3<scalar_t> _T(single_T.data<scalar_t>());
            
            _T = getRandomTranslation<scalar_t>(volume_size, _a, _b);
        }));
    }
}
