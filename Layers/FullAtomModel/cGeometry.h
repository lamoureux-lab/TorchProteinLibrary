#ifndef CGEOMETRY_H_
#define CGEOMETRY_H_

class cGeometry{
    public:
        double zero_const;
        //backbone angles
        double omega_const, C_N_CA_angle, N_CA_C_angle, CA_C_N_angle, CA_C_O_angle; 
        //backbone distances
        double R_CA_C, R_C_N, R_N_CA, R_C_O;
        double C_CA_CB_angle, N_C_CA_CB_diangle;

        //C-beta atom
        double R_CA_CB;

        cGeometry();
        ~cGeometry();

        void gly();
        void ala();
};

#endif