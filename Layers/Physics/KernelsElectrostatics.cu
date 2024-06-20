#include <KernelsElectrostatics.h>
#include <stdio.h>

#include <cusp/dia_matrix.h>
#include <cusp/monitor.h>
#include <cusp/precond/diagonal.h>
#include <cusp/krylov/cg.h>

#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>


__global__ void partialSumFaces(    float* coords, float* assigned_params, int num_atoms, float* volume, 
                                    int box_size, float res, float ion_size, float wat_size, float asigma, uint d){

    int vol_index = threadIdx.x;
	float *single_volume = volume + vol_index * box_size*box_size*box_size;

    float cell_x, cell_y, cell_z;
    float shift_cell_x=0.0, shift_cell_y=0.0, shift_cell_z=0.0;
    float add_sigma = 0.0;
    switch(vol_index){
        case 0:
            shift_cell_x = 0.5;
            add_sigma = wat_size;
            break;
        case 1:
            shift_cell_y = 0.5;
            add_sigma = wat_size;
            break;
        case 2:
            shift_cell_z = 0.5;
            add_sigma = wat_size;
            break;
        case 3:
            add_sigma = ion_size;
            break;
    }
    long cell_idx;

    float x, y, z;
    int x_i, y_i, z_i;
    float r2;


	for(int idx = 0; idx<num_atoms; idx++){
        float sigma2 = asigma*(assigned_params[idx] + add_sigma);
        sigma2 *= sigma2;
		x = coords[3*idx + 0];
		y = coords[3*idx + 1];
		z = coords[3*idx + 2];
		
        x_i = floor(x/res);
		y_i = floor(y/res);
		z_i = floor(z/res);
		for(int i=x_i-d; i<=(x_i+d);i++){
			for(int j=y_i-d; j<=(y_i+d);j++){
				for(int k=z_i-d; k<=(z_i+d);k++){
					if( (i>=0 && i<box_size) && (j>=0 && j<box_size) && (k>=0 && k<box_size) ){
						cell_idx = k + j*box_size + i*box_size*box_size;
						cell_x = i*res + shift_cell_x*res;
                        cell_y = j*res + shift_cell_y*res;
                        cell_z = k*res + shift_cell_z*res;
                        r2 = (x-cell_x)*(x-cell_x) + (y-cell_y)*(y-cell_y) + (z-cell_z)*(z-cell_z);
						single_volume[cell_idx] *= 1.0 - exp(-r2/sigma2);
					}
				}
			}
		}
	}
}

__global__ void sumCells(   float* coords, float* assigned_params, int num_atoms, float* volume, 
                            int box_size, float res){
	for(int idx = 0; idx<num_atoms; idx++){
		float x = coords[3*idx + 0],
            y = coords[3*idx + 1],
            z = coords[3*idx + 2];
		int x_i = floor(x/res);
		int y_i = floor(y/res);
		int z_i = floor(z/res);
        int cell_idx = z_i + y_i*box_size + x_i*box_size*box_size;
        // printf("%f, %f, %f, %d, %f\n", x, y, z, cell_idx, assigned_params[idx]);
		volume[cell_idx] += assigned_params[idx];
	}
}

void gpu_computePartialSumFaces(	float *coords,
                                    float *assigned_params,
                                    int num_atoms, 
                                    float *volume,
                                    int box_size,
                                    float res,
                                    float ion_size, float wat_size, float asigma,
                                    uint d){
	partialSumFaces<<<1, 4>>>(coords, assigned_params, num_atoms, volume, box_size, res, ion_size, wat_size, asigma, d);
}

void gpu_computeSumCells(	float *coords,
                            float *assigned_params,
                            int num_atoms, 
                            float *volume,
                            int box_size,
                            float res){
	sumCells<<<1, 1>>>(coords, assigned_params, num_atoms, volume, box_size, res);
}

struct saxpy_functor
{
    const float mul;
    saxpy_functor(float _mul):mul(_mul){}

    __host__ __device__
    float operator()(const float& x, const float& y) const
    { 
        return mul * x + y;
    }
};

struct fill_functor
{
    const size_t surf_size, box_size;
    const float val;
    fill_functor(size_t _surf_size, size_t _box_size, float _val)
    :surf_size(_surf_size), box_size(_box_size), val(_val){}

    __host__ __device__
    float operator()(const size_t& idx, const float& old_val) const
    { 
        int x = floor(idx/surf_size);
        int y = floor((idx - x*surf_size)/box_size);
        int z = idx - x*surf_size - y*box_size;
        
        if( (x==0||x==(box_size-1)) || (y==0||y==(box_size-1)) || (z==0||z==(box_size-1))){
            return val;
        }else{
            return old_val;
        }
    }
};


void gpu_computePhi( float *Q, float *Eps, float *Phi, size_t box_size, float res, float kappa02){
    size_t surf_size = box_size*box_size; 
    size_t vol_size = box_size*box_size*box_size;
    size_t num_nonzero = 7*vol_size - 2*surf_size - 2*box_size - 2;

    cusp::dia_matrix<size_t, float, cusp::device_memory> A(vol_size, vol_size, num_nonzero, 7);
    cusp::array1d<float, cusp::device_memory> phi(vol_size, 0.0);
    cusp::array1d<float, cusp::device_memory> q(vol_size, 0.0);
    A.diagonal_offsets[0] = -box_size*box_size;
    A.diagonal_offsets[1] = -box_size;
    A.diagonal_offsets[2] = -1;
    A.diagonal_offsets[3] = 0;
    A.diagonal_offsets[4] = 1;
    A.diagonal_offsets[5] = box_size;
    A.diagonal_offsets[6] = box_size*box_size;
    
    thrust::fill(A.values.values.begin(), A.values.values.end(), 0.0);
    
    thrust::device_ptr<float> ei_begin(Eps);
    thrust::device_ptr<float> ej_begin(Eps + vol_size);
    thrust::device_ptr<float> ek_begin(Eps + 2*vol_size);
    thrust::device_ptr<float> lambda_begin(Eps + 3*vol_size);
    thrust::device_ptr<float> q_begin(Q);
    thrust::device_ptr<float> Phi_begin(Phi);
    
    //Lower diagonals
    thrust::transform(ei_begin, ei_begin + vol_size - surf_size, A.values.column(0).begin() + surf_size, 
                        A.values.column(0).begin() + surf_size, saxpy_functor(-res));
    thrust::transform(ej_begin, ej_begin + vol_size - box_size, A.values.column(1).begin() + box_size, 
                        A.values.column(1).begin() + box_size, saxpy_functor(-res));
    thrust::transform(ek_begin, ek_begin + vol_size - 1, A.values.column(2).begin() + 1, 
                        A.values.column(2).begin() + 1, saxpy_functor(-res));

    //Upper diagonals
    thrust::transform(ek_begin, ek_begin + vol_size - 1, A.values.column(4).begin(), 
                        A.values.column(4).begin(), saxpy_functor(-res));
    thrust::transform(ej_begin, ej_begin + vol_size - box_size, A.values.column(5).begin(), 
                        A.values.column(5).begin(), saxpy_functor(-res));
    thrust::transform(ei_begin, ei_begin + vol_size - surf_size, A.values.column(6).begin(), 
                        A.values.column(6).begin(), saxpy_functor(-res));
    
    //Diagonal
    thrust::transform(ei_begin, ei_begin + vol_size, A.values.column(3).begin(), 
                        A.values.column(3).begin(), saxpy_functor(res));
    thrust::transform(ej_begin, ej_begin + vol_size, A.values.column(3).begin(), 
                        A.values.column(3).begin(), saxpy_functor(res));
    thrust::transform(ek_begin, ek_begin + vol_size, A.values.column(3).begin(), 
                        A.values.column(3).begin(), saxpy_functor(res));
    
    //diagonal shifted
    thrust::transform(ei_begin, ei_begin + vol_size - surf_size, A.values.column(3).begin() + surf_size, 
                        A.values.column(3).begin() + surf_size, saxpy_functor(res));
    thrust::transform(ej_begin, ej_begin + vol_size - box_size, A.values.column(3).begin() + box_size,
                        A.values.column(3).begin() + box_size, saxpy_functor(res));
    thrust::transform(ek_begin, ek_begin + vol_size - 1, A.values.column(3).begin() + 1,
                        A.values.column(3).begin() + 1, saxpy_functor(res));
    
    //ionic term
    thrust::transform(lambda_begin, lambda_begin + vol_size, A.values.column(3).begin(), 
                        A.values.column(3).begin(), saxpy_functor(kappa02*res*res*res));

    //boundary conditions (A_bound = 1)
    thrust::counting_iterator<size_t> first(0);
    thrust::counting_iterator<size_t> last(vol_size);
    thrust::transform(first, last, A.values.column(3).begin(), A.values.column(3).begin(), fill_functor(surf_size, box_size, 1.0f));
    
    thrust::transform(first, last, A.values.column(0).begin(), A.values.column(0).begin(), fill_functor(surf_size, box_size, 0.0f));
    thrust::transform(first, last, A.values.column(1).begin(), A.values.column(1).begin(), fill_functor(surf_size, box_size, 0.0f));
    thrust::transform(first, last, A.values.column(2).begin(), A.values.column(2).begin(), fill_functor(surf_size, box_size, 0.0f));
    
    thrust::transform(first, last, A.values.column(4).begin(), A.values.column(4).begin(), fill_functor(surf_size, box_size, 0.0f));
    thrust::transform(first, last, A.values.column(5).begin(), A.values.column(5).begin(), fill_functor(surf_size, box_size, 0.0f));
    thrust::transform(first, last, A.values.column(6).begin(), A.values.column(6).begin(), fill_functor(surf_size, box_size, 0.0f));
    

    //charge
    thrust::transform(q_begin, q_begin + vol_size, q.begin(), q.begin(), saxpy_functor(1.0));

    cusp::monitor<float> monitor(q, 1000, 1e-3, 0.0, false);
    // monitor.set_verbose();
    cusp::precond::diagonal<float, cusp::device_memory> M(A);
    cusp::krylov::cg(A, phi, q, monitor);
    // monitor.print();

    thrust::copy(phi.begin(), phi.end(), Phi);

       
}