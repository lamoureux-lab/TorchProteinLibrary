#include <cGeometry.h>
#include <math.h>

#define KAPPA1 (3.14159 - 1.9391)
#define KAPPA2 (3.14159 - 2.061)
#define KAPPA3 (3.14159 -2.1186)
#define OMEGACIS -3.1318

cGeometry::cGeometry(){
     //backbone angles
    C_N_CA_angle = (M_PI - 1.9391);
    N_CA_C_angle = (M_PI - 2.061);
    CA_C_N_angle = (M_PI - 2.1186);
    CA_C_O_angle = (M_PI - 2.1033);
    omega_const = -3.1318;
    //backbone distatnces
    R_CA_C = 1.525;
    R_C_N = 1.330;
    R_N_CA = 1.460;

    //C-beta
    R_CA_CB = 1.52;
    C_CA_CB_angle = -(M_PI - 1.9111);

    //OG serine
    R_CB_OG = 1.417;
    CA_CB_OG_angle = M_PI*110.773/180.0;
    //SG cysteine
    R_CB_SG = 1.808;
    CA_CB_SG_angle = M_PI*113.8169/180.0;
    //Valine CG1 and CG2
    R_CB_CG = 1.527;
    CA_CB_CG1_angle =  M_PI*110.7/180.0;
    CG1_CB_CG2_angle = 1.88444687881;
    //Isoleucine
}

cGeometry::~cGeometry(){
    
}

void cGeometry::gly(){
    //backbone angles
    C_N_CA_angle = (3.14159 - 1.9391);
    N_CA_C_angle = (3.14159 - 2.061);
    CA_C_N_angle = (3.14159 - 2.1186);
    CA_C_O_angle = (3.14159 - 2.1033);
    omega_const = -3.1318;
    //backbone distatnces
    R_CA_C = 1.525;
    R_C_N = 1.330;
    R_N_CA = 1.460;
}

void cGeometry::ala(){
    //backbone angles
    C_N_CA_angle = (3.14159 - 1.9391);
    N_CA_C_angle = (3.14159 - 2.061);
    CA_C_N_angle = (3.14159 - 2.1186);
    CA_C_O_angle = (3.14159 - 2.1033);
    omega_const = -3.1318;
    //backbone distatnces
    R_CA_C = 1.525;
    R_C_N = 1.330;
    R_N_CA = 1.460;

    //C-beta
    R_CA_CB = 1.52;
    C_CA_CB_angle = -(3.14159 - 1.9111);//(3.14159 - 1.9111);
    N_C_CA_CB_diangle = 0.0;//(3.14159 - 2.1413);
}