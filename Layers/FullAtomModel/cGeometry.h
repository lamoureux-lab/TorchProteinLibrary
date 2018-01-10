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

        //Serine
        double CA_CB_OG_angle;
        double R_CB_OG;
        //Cysteine
        double CA_CB_SG_angle;
        double R_CB_SG;
        //Valine
        double R_CB_CG;
        double CG1_CB_CG2_angle, CA_CB_CG1_angle;
        //Isoleucine
        double R_CG1_CD1;
        double CB_CG1_CD1_angle;
        //Threonine
        double R_CB_OG1;
        double CA_CB_OG1_angle, OG1_CB_CG2_angle;
        //Arginine
        double R_C_C;
        double C_C_C_angle;
        double R_CD_NE, R_NE_CZ, R_CZ_NH;
        double CA_CB_CG_angle, CB_CG_CD_angle, CG_CD_NE_angle;
        double CD_NE_CZ_angle, NE_CZ_NH_angle;
        //Lysine
        double R_CD_CE, R_CE_NZ;
        double CD_CE_NZ_angle;
        //Aspartic acid
        double R_CG_OD;
        double CB_CG_OD_angle, OD1_CG_OD2_angle;
        //Asparagine
        double R_CG_OD1, R_CG_ND2;
        double CB_CG_OD1_angle, CB_CG_ND2_angle; 
        //Methionine
        double R_CG_SD, R_SD_CE;
        double CB_CG_SD_angle, CG_SD_CE_angle;

        cGeometry();
        ~cGeometry();

        void gly();
        void ala();
};

#endif