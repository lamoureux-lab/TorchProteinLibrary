#include "cConformation.h"
#include <iostream>
#include <string>
#include "nUtil.h"
#include "coordsTransform_interface.h"


void CoordsTranslate_forward(   at::Tensor input_coords, 
                                at::Tensor output_coords,
                                at::Tensor T,
                                at::Tensor num_atoms
                                ){

    if( input_coords.dtype() != at::kDouble || output_coords.dtype() != at::kDouble || T.dtype() != at::kDouble 
    || num_atoms.dtype() != at::kInt){
        throw("Incorrect tensor types");
    }
    if(input_coords.ndimension() != 2){
        throw("Incorrect input ndim");
    }
    int batch_size = input_coords.size(0);
    auto num_at = num_atoms.accessor<int,1>();
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_input_coords = input_coords[i];
        at::Tensor single_output_coords = output_coords[i];
        auto aT = T.accessor<double,2>();
        cVector3 translation(aT[i][0], aT[i][1], aT[i][2]);
        ProtUtil::translate(single_input_coords, translation, single_output_coords, num_at[i]);
    }
}
void CoordsRotate_forward(  at::Tensor input_coords, 
                            at::Tensor output_coords,
                            at::Tensor R,
                            at::Tensor num_atoms
                            ){
    if( input_coords.dtype() != at::kDouble || output_coords.dtype() != at::kDouble || R.dtype() != at::kDouble 
    || num_atoms.dtype() != at::kInt){
        throw("Incorrect tensor types");
    }
    if(input_coords.ndimension() != 2){
        throw("Incorrect input ndim");
    }
    
    int batch_size = input_coords.size(0);
    auto num_at = num_atoms.accessor<int,1>();
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_input_coords = input_coords[i];
        at::Tensor single_output_coords = output_coords[i];
        at::Tensor single_R = R[i];
        
        cMatrix33 _R = ProtUtil::tensor2Matrix33(single_R);
        ProtUtil::rotate(single_input_coords, _R, single_output_coords, num_at[i]);
    }
}
void CoordsRotate_backward( at::Tensor grad_output_coords, 
                            at::Tensor grad_input_coords,
                            at::Tensor R,
                            at::Tensor num_atoms){
    if( grad_output_coords.dtype() != at::kDouble || grad_input_coords.dtype() != at::kDouble || R.dtype() != at::kDouble 
    || num_atoms.dtype() != at::kInt){
        throw("Incorrect tensor types");
    }
    if(grad_output_coords.ndimension() != 2){
        throw("Incorrect input ndim");
    }
    
    
    int batch_size = grad_output_coords.size(0);
    auto num_at = num_atoms.accessor<int,1>();
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_grad_output_coords = grad_output_coords[i];
        at::Tensor single_grad_input_coords = grad_input_coords[i];
        at::Tensor single_R = R[i];
        
        cMatrix33 _R = ProtUtil::tensor2Matrix33(single_R);
        _R = _R.getTranspose();
        ProtUtil::rotate(single_grad_output_coords, _R, single_grad_input_coords, num_at[i]);
    }
}
void getBBox(   at::Tensor input_coords,
                at::Tensor a, at::Tensor b,
                at::Tensor num_atoms){
    if( input_coords.dtype() != at::kDouble || a.dtype() != at::kDouble || b.dtype() != at::kDouble 
    || num_atoms.dtype() != at::kInt){
        throw("Incorrect tensor types");
    }
    if(input_coords.ndimension() != 2){
        throw("Incorrect input ndim");
    }
    
    int batch_size = input_coords.size(0);
    auto num_at = num_atoms.accessor<int,1>();
    #pragma omp parallel for num_threads(10)
    for(int i=0; i<batch_size; i++){
        at::Tensor single_input_coords = input_coords[i];
        at::Tensor single_a = a[i];
        at::Tensor single_b = b[i];
        
        cVector3 va(single_a.data<double>());
        cVector3 vb(single_b.data<double>());
        ProtUtil::computeBoundingBox(single_input_coords, num_at[i], va, vb);
    }
}
void getRandomRotation( at::Tensor R ){
    if( R.dtype() != at::kDouble ){
        throw("Incorrect tensor types");
    }
    if(R.ndimension() != 3){
        throw("Incorrect input ndim");
    }

    int batch_size = R.size(0);
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_R = R[i];
        cMatrix33 rnd_R = ProtUtil::getRandomRotation();
        ProtUtil::matrix2Tensor(rnd_R, single_R);                
    }
}
void getRotation( at::Tensor R, at::Tensor u ){
    if( R.dtype() != at::kDouble || u.dtype() != at::kDouble ){
        throw("Incorrect tensor types");
    }
    if(R.ndimension() != 3 || u.ndimension() !=2 ){
        throw("Incorrect input ndim");
    }

    int batch_size = R.size(0);
    auto param = u.accessor<double,2>();

    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_R = R[i];
        cMatrix33 R = ProtUtil::getRotation(param[i][0], param[i][1], param[i][2]);
        ProtUtil::matrix2Tensor(R, single_R);
    }
}
void getRandomTranslation( at::Tensor T, at::Tensor a, at::Tensor b, int volume_size){
    if( T.dtype() != at::kDouble || a.dtype() != at::kDouble || b.dtype() != at::kDouble){
        throw("Incorrect tensor types");
    }
    if(T.ndimension() != 2){
        throw("Incorrect input ndim");
    }
    
    int batch_size = T.size(0);
    #pragma omp parallel for
    for(int i=0; i<batch_size; i++){
        at::Tensor single_T = T[i];
        at::Tensor single_a = a[i];
        at::Tensor single_b = b[i];
                
        cVector3 _a(single_a.data<double>());
        cVector3 _b(single_b.data<double>());
        cVector3 _T(single_T.data<double>());
        
        _T = ProtUtil::getRandomTranslation(volume_size, _a, _b);
    }
}
