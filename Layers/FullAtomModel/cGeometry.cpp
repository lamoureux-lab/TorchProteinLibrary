#include <cGeometry.h>

#define KAPPA1 (3.14159 - 1.9391)
#define KAPPA2 (3.14159 - 2.061)
#define KAPPA3 (3.14159 -2.1186)
#define OMEGACIS -3.1318

cGeometry::cGeometry(){
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
    C_CA_CB_angle = -(3.14159 - 1.9111);
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