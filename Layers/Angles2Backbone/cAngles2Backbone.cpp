#include <iostream>
#include <fstream>
#include <exception>
#include <limits>
#include <math.h>
#include <set>

#include "cAngles2Backbone.h"
#include "cTensorProteinCUDAKernels.h"

cAngles2Backbone::cAngles2Backbone(THCState *state, THFloatTensor *input_angles){
    this->angles_length = input_angles->size[1];
    this->num_atoms_calpha = angles_length;
    this->num_atoms_c = angles_length;
    this->num_atoms_n = angles_length;

    this->d_phi = THCudaTensor_newWithSize1d(state, angles_length);
	this->d_psi = THCudaTensor_newWithSize1d(state, angles_length);
    this->d_dphi = THCudaTensor_newWithSize1d(state, angles_length);
	this->d_dpsi = THCudaTensor_newWithSize1d(state, angles_length);
    this->d_A = THCudaTensor_newWithSize1d(state, 3*16*angles_length);

    this->d_r_calpha = THCudaTensor_newWithSize1d(state, 3*num_atoms_calpha);
    this->d_r_c = THCudaTensor_newWithSize1d(state, 3*num_atoms_c);
    this->d_r_n = THCudaTensor_newWithSize1d(state, 3*num_atoms_n);

    this->d_drdphi = THCudaTensor_newWithSize1d(state, 3*num_atoms_calpha*angles_length);
    this->d_drdpsi = THCudaTensor_newWithSize1d(state, 3*num_atoms_calpha*angles_length);

    this->c_phi = THFloatTensor_new();
	this->c_psi = THFloatTensor_new();
    this->c_r_calpha = THFloatTensor_newWithSize1d(3*num_atoms_calpha);
    this->c_r_c = THFloatTensor_newWithSize1d(3*num_atoms_c);
    this->c_r_n = THFloatTensor_newWithSize1d(3*num_atoms_n);

    
    THFloatTensor_select(this->c_phi, input_angles, 0, 0);
	THFloatTensor_select(this->c_psi, input_angles, 0, 1);

    toGPUTensor(state, c_phi, d_phi);
    toGPUTensor(state, c_psi, d_psi);
    
    std::cout<<"Init finished"<<std::endl;

}
cAngles2Backbone::~cAngles2Backbone(){
    THCudaTensor_free(state, this->d_phi);
	THCudaTensor_free(state, this->d_psi);
    THCudaTensor_free(state, this->d_dphi);
	THCudaTensor_free(state, this->d_dpsi);
    THCudaTensor_free(state, this->d_A);
    THCudaTensor_free(state, this->d_r_calpha);
    THCudaTensor_free(state, this->d_r_c);
    THCudaTensor_free(state, this->d_r_n);

    THFloatTensor_free(this->c_phi);
	THFloatTensor_free(this->c_psi);
    THFloatTensor_free(this->c_r_calpha);
    THFloatTensor_free(this->c_r_c);
    THFloatTensor_free(this->c_r_n);
    


}
void cAngles2Backbone::computeForward(){
    // Forward pass angles -> coordinates
	cpu_computeCoordinates(	THCudaTensor_data(state, this->d_phi), 
							THCudaTensor_data(state, this->d_psi), 
							THCudaTensor_data(state, this->d_r_calpha), 
                            THCudaTensor_data(state, this->d_r_c),
                            THCudaTensor_data(state, this->d_r_n),
							THCudaTensor_data(state, this->d_A), 
							this->angles_length);
}   

void cAngles2Backbone::computeForwardCalpha(){
    // Forward pass angles -> coordinates
	cpu_computeCoordinatesCalpha(	THCudaTensor_data(state, this->d_phi), 
                                    THCudaTensor_data(state, this->d_psi), 
                                    THCudaTensor_data(state, this->d_r_calpha),
                                    THCudaTensor_data(state, this->d_A), 
                                    this->angles_length);
}   


void cAngles2Backbone::computeBackwardCalpha( THCudaTensor *gradInput,            //output gradient of the input angles
                                        THCudaTensor *gradOutput_coords ){  //input gradient of the coordinates
                                    
    // Computing the derivative vectors of coordinates wrt alpha and beta
    cpu_computeDerivativesCalpha(	THCudaTensor_data(state, this->d_phi), 
                                    THCudaTensor_data(state, this->d_psi), 
                                    THCudaTensor_data(state, this->d_drdphi), 
                                    THCudaTensor_data(state, this->d_drdpsi), 
                                    THCudaTensor_data(state, this->d_A), 
                                    this->angles_length);
    // Backward pass from gradients of coordinates to the gradients of angles
	THCudaTensor_select(state, d_dphi, gradInput, 0, 0);
	THCudaTensor_select(state, d_dpsi, gradInput, 0, 1);
	cpu_backwardFromCoordsCalpha( THCudaTensor_data(state, this->d_dphi), 
							THCudaTensor_data(state, this->d_dpsi), 
							THCudaTensor_data(state, gradOutput_coords), //1d Tensor: 3 x number of atoms 
							THCudaTensor_data(state, this->d_drdphi), 
							THCudaTensor_data(state, this->d_drdpsi), 
							this->angles_length);
}

void cAngles2Backbone::display(){
     
    computeForward();
    toCPUTensor(state, d_r_calpha, c_r_calpha);
    toCPUTensor(state, d_r_c, c_r_c);
    toCPUTensor(state, d_r_n, c_r_n);
    
    display_backbone();
    
}

void cAngles2Backbone::display_Calpha(){
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);
    glPointSize(5);
	glBegin(GL_POINTS);
		glColor3f(1.0,0.0,0.0);
        for(int i=0; i<num_atoms_calpha; i++){
            float x = THFloatTensor_get1d(c_r_calpha, 3*i);
            float y = THFloatTensor_get1d(c_r_calpha, 3*i+1);
            float z = THFloatTensor_get1d(c_r_calpha, 3*i+2);
            glVertex3f(x,y,z);
        }
    glEnd();

    glLineWidth(3);
    glBegin(GL_LINES);
        glColor3f(1.0,1.0,1.0);
        for(int i=0; i<(num_atoms_calpha-1); i++){
            float x0 = THFloatTensor_get1d(c_r_calpha, 3*i);
            float y0 = THFloatTensor_get1d(c_r_calpha, 3*i+1);
            float z0 = THFloatTensor_get1d(c_r_calpha, 3*i+2);
            float x1 = THFloatTensor_get1d(c_r_calpha, 3*(i+1));
            float y1 = THFloatTensor_get1d(c_r_calpha, 3*(i+1)+1);
            float z1 = THFloatTensor_get1d(c_r_calpha, 3*(i+1)+2);
            glVertex3f(x0,y0,z0);
            glVertex3f(x1,y1,z1);
        }
    glEnd();
    glPopAttrib();
}

void cAngles2Backbone::display_backbone(){
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);
    glPointSize(5);
	glBegin(GL_POINTS);
		glColor3f(1.0,0.0,0.0);
        for(int i=0; i<num_atoms_calpha; i++){
            float x = THFloatTensor_get1d(c_r_calpha, 3*i);
            float y = THFloatTensor_get1d(c_r_calpha, 3*i+1);
            float z = THFloatTensor_get1d(c_r_calpha, 3*i+2);
            glVertex3f(x,y,z);
        }
        glColor3f(0.0,1.0,0.0);
        for(int i=0; i<num_atoms_c; i++){
            float x = THFloatTensor_get1d(c_r_c, 3*i);
            float y = THFloatTensor_get1d(c_r_c, 3*i+1);
            float z = THFloatTensor_get1d(c_r_c, 3*i+2);
            glVertex3f(x,y,z);
        }
        glColor3f(0.0,0.0,1.0);
        for(int i=0; i<num_atoms_n; i++){
            float x = THFloatTensor_get1d(c_r_n, 3*i);
            float y = THFloatTensor_get1d(c_r_n, 3*i+1);
            float z = THFloatTensor_get1d(c_r_n, 3*i+2);
            glVertex3f(x,y,z);
        }
    glEnd();

    glLineWidth(3);
    glBegin(GL_LINES);
        glColor3f(1.0,1.0,1.0);
        for(int i=0; i<(num_atoms_calpha); i++){
            float x0 = THFloatTensor_get1d(c_r_calpha, 3*i);
            float y0 = THFloatTensor_get1d(c_r_calpha, 3*i+1);
            float z0 = THFloatTensor_get1d(c_r_calpha, 3*i+2);
            float x1 = THFloatTensor_get1d(c_r_c, 3*i);
            float y1 = THFloatTensor_get1d(c_r_c, 3*i+1);
            float z1 = THFloatTensor_get1d(c_r_c, 3*i+2);
            float x2 = THFloatTensor_get1d(c_r_n, 3*i);
            float y2 = THFloatTensor_get1d(c_r_n, 3*i+1);
            float z2 = THFloatTensor_get1d(c_r_n, 3*i+2);
            
            
            glVertex3f(x2,y2,z2);
            glVertex3f(x0,y0,z0);

            glVertex3f(x0,y0,z0);
            glVertex3f(x1,y1,z1);

            if ( (i+1)<num_atoms_calpha){
                float x3 = THFloatTensor_get1d(c_r_n, 3*(i+1));
                float y3 = THFloatTensor_get1d(c_r_n, 3*(i+1)+1);
                float z3 = THFloatTensor_get1d(c_r_n, 3*(i+1)+2);
                glVertex3f(x1,y1,z1);
                glVertex3f(x3,y3,z3);
            }
        }
	glColor3f(1.0,0.0,0.0);
        for(int i=0; i<(num_atoms_calpha); i++){
            float x0 = THFloatTensor_get1d(c_r_calpha, 3*i);
            float y0 = THFloatTensor_get1d(c_r_calpha, 3*i+1);
            float z0 = THFloatTensor_get1d(c_r_calpha, 3*i+2);           
            
            if ( (i+1)<num_atoms_calpha){
                float x3 = THFloatTensor_get1d(c_r_calpha, 3*(i+1));
                float y3 = THFloatTensor_get1d(c_r_calpha, 3*(i+1)+1);
                float z3 = THFloatTensor_get1d(c_r_calpha, 3*(i+1)+2);
                glVertex3f(x0,y0,z0);
                glVertex3f(x3,y3,z3);
            }
        }
    glEnd();
    glPopAttrib();
}
