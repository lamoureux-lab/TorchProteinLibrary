#ifndef CGEOMETRY_H_
#define CGEOMETRY_H_

template <typename T>
class cGeometry{
    public:
        T zero_const;
        //backbone angles
        T omega_const, C_N_CA_angle, N_CA_C_angle, CA_C_N_angle, CA_C_O_angle; 
        
        //backbone distances
        T R_CA_C, R_C_N, R_N_CA, R_C_O;
        T C_CA_CB_angle, N_C_CA_CB_diangle, correction_angle;

        //C-beta atom
        T R_CA_CB;

        //Serine
        T CA_CB_OG_angle;
        T R_CB_OG;
        //Cysteine
        T CA_CB_SG_angle;
        T R_CB_SG;
        //Valine
        T R_CB_CG;
        T CG1_CB_CG2_angle, CA_CB_CG1_angle;
        //Isoleucine
        T R_CG1_CD1;
        T CB_CG1_CD1_angle;
        //Threonine
        T R_CB_OG1;
        T CA_CB_OG1_angle, OG1_CB_CG2_angle;
        //Arginine
        T R_C_C;
        T C_C_C_angle;
        T R_CD_NE, R_NE_CZ, R_CZ_NH;
        T CA_CB_CG_angle, CB_CG_CD_angle, CG_CD_NE_angle;
        T CD_NE_CZ_angle, NE_CZ_NH_angle;
        //Lysine
        T R_CD_CE, R_CE_NZ;
        T CD_CE_NZ_angle;
        //Aspartic acid
        T R_CG_OD;
        T CB_CG_OD_angle, OD1_CG_OD2_angle;
        //Asparagine
        T R_CG_OD1, R_CG_ND2;
        T CB_CG_OD1_angle, CB_CG_ND2_angle; 
        //Methionine
        T R_CG_SD, R_SD_CE;
        T CB_CG_SD_angle, CG_SD_CE_angle;
        //Histidine
        T R_CG_CD2, R_CG_ND1, R_CG_NE2, R_CG_CE1;
        T CB_CG_CD2_angle, CB_CG_ND1_angle, CB_CG_NE2_angle, CB_CG_CE1_angle;

        cGeometry();
        ~cGeometry();

        void gly();
        void ala();
};

#endif