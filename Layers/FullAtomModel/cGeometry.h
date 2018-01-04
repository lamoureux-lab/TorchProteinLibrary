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

        //Serine OG
        double CA_CB_OG_angle;
        double R_CB_OG;
        //Cysteine SG
        double CA_CB_SG_angle;
        double R_CB_SG;
        //Valine
        double R_CB_CG;
        double CA_CB_CG1_angle, CA_CB_CG2_angle;
        

        cGeometry();
        ~cGeometry();

        void gly();
        void ala();
};

#endif