#include <THC/THC.h>
#include "cCoords2RMSD.h"
#include <iostream>

extern THCState *state;
float R = 3.8;
extern "C" {
    int Coords2RMSD_forward(    THCudaTensor *input, THCudaTensor *target, THCudaTensor *output, THIntTensor *angles_length,
                                THCudaTensor *re_coordinates_src, 
                                THCudaTensor *re_coordinates_dst,
                                THCudaTensor *U_coordinates_src,
                                THCudaTensor *Ut_coordinates_dst,
                                THCudaTensor *rot_mat_t
                            ){
        if(input->nDimension == 1){
            cCoords2RMSD c2r(state,
                    re_coordinates_src, 
                    re_coordinates_dst,
                    U_coordinates_src,
                    Ut_coordinates_dst,
                    rot_mat_t,
                    THIntTensor_get1d(angles_length, 0));
            float rmsd = c2r.computeForward(input, target);
            THCudaTensor_set1d(state, output, 0, rmsd);
        }else{
            // #pragma omp parallel for num_threads(10)
            for(int i=0; i<input->size[0]; i++){
                //inputs and outputs
                THCudaTensor *single_input, *single_target;
                single_input = THCudaTensor_new(state);
	            single_target = THCudaTensor_new(state);
                THCudaTensor_select(state, single_input, input, 0, i);
                THCudaTensor_select(state, single_target, target, 0, i);
                //params
                THCudaTensor *s_re_coordinates_src, *s_re_coordinates_dst, *s_U_coordinates_src, *s_Ut_coordinates_dst, *s_rot_mat_t;
                s_re_coordinates_src = THCudaTensor_new(state);
	            s_re_coordinates_dst = THCudaTensor_new(state);
                s_U_coordinates_src = THCudaTensor_new(state);
                s_Ut_coordinates_dst = THCudaTensor_new(state);
                s_rot_mat_t = THCudaTensor_new(state);
                THCudaTensor_select(state, s_re_coordinates_src, re_coordinates_src, 0, i);
                THCudaTensor_select(state, s_re_coordinates_dst, re_coordinates_dst, 0, i);
                THCudaTensor_select(state, s_U_coordinates_src, U_coordinates_src, 0, i);
                THCudaTensor_select(state, s_Ut_coordinates_dst, Ut_coordinates_dst, 0, i);
                THCudaTensor_select(state, s_rot_mat_t, rot_mat_t, 0, i);
                
                cCoords2RMSD c2r(   state, s_re_coordinates_src, s_re_coordinates_dst, s_U_coordinates_src,
                                    s_Ut_coordinates_dst, s_rot_mat_t, THIntTensor_get1d(angles_length, i));
                float rmsd = c2r.computeForward(single_input, single_target);
                THCudaTensor_set1d(state, output, i, rmsd);

                THCudaTensor_free(state, single_input);
                THCudaTensor_free(state, single_target);
                THCudaTensor_free(state, s_re_coordinates_src);
                THCudaTensor_free(state, s_re_coordinates_dst);
                THCudaTensor_free(state, s_U_coordinates_src);
                THCudaTensor_free(state, s_Ut_coordinates_dst);
                THCudaTensor_free(state, s_rot_mat_t);
                
            }
        }
    }
    int Coords2RMSD_backward( THCudaTensor *gradInput,  THCudaTensor *gradOutput, THIntTensor *angles_length,
                            THCudaTensor *re_coordinates_src,
                            THCudaTensor *re_coordinates_dst,
                            THCudaTensor *Ut_coordinates_dst,
                            THCudaTensor *rot_mat_t        
                            ){
        if(re_coordinates_src->nDimension == 1){
            cCoords2RMSD c2r(state,
                    re_coordinates_src, 
                    re_coordinates_dst,
                    re_coordinates_dst,
                    Ut_coordinates_dst,
                    rot_mat_t,
                    THIntTensor_get1d(angles_length, 0));
            c2r.computeBackward(gradInput, THCudaTensor_get1d(state, gradOutput, 0));
        }else{
            // #pragma omp parallel for num_threads(10)
            for(int i=0; i<gradInput->size[0]; i++){
                //inputs and outputs
                THCudaTensor *single_gradInput;
                single_gradInput = THCudaTensor_new(state);
                THCudaTensor_select(state, single_gradInput, gradInput, 0, i);
                
                //params
                THCudaTensor *s_re_coordinates_src, *s_re_coordinates_dst, *s_U_coordinates_src, *s_Ut_coordinates_dst, *s_rot_mat_t;
                s_re_coordinates_src = THCudaTensor_new(state);
	            s_re_coordinates_dst = THCudaTensor_new(state);
                s_U_coordinates_src = THCudaTensor_new(state);
                s_Ut_coordinates_dst = THCudaTensor_new(state);
                s_rot_mat_t = THCudaTensor_new(state);
                THCudaTensor_select(state, s_re_coordinates_src, re_coordinates_src, 0, i);
                THCudaTensor_select(state, s_re_coordinates_dst, re_coordinates_dst, 0, i);
                THCudaTensor_select(state, s_Ut_coordinates_dst, Ut_coordinates_dst, 0, i);
                THCudaTensor_select(state, s_rot_mat_t, rot_mat_t, 0, i);
                
                cCoords2RMSD c2r(   state, s_re_coordinates_src, s_re_coordinates_dst, s_U_coordinates_src,
                                    s_Ut_coordinates_dst, s_rot_mat_t, THIntTensor_get1d(angles_length, i));
                c2r.computeBackward(single_gradInput, THCudaTensor_get1d(state, gradOutput, i));

                THCudaTensor_free(state, single_gradInput);
                THCudaTensor_free(state, s_re_coordinates_src);
                THCudaTensor_free(state, s_re_coordinates_dst);
                THCudaTensor_free(state, s_U_coordinates_src);
                THCudaTensor_free(state, s_Ut_coordinates_dst);
                THCudaTensor_free(state, s_rot_mat_t);
            }
        }
    }


}