#include <cGeometry.h>
#include <math.h>

#define KAPPA1 (3.14159 - 1.9391)
#define KAPPA2 (3.14159 - 2.061)
#define KAPPA3 (3.14159 -2.1186)
#define OMEGACIS -3.1318
template <typename T> cGeometry<T>::cGeometry(){
     //backbone angles [Original TPL angles: from Peptide Builder]
//    C_N_CA_angle = (M_PI - 1.9391);
//    N_CA_C_angle = (M_PI - 2.061);
//    CA_C_N_angle = (M_PI - 2.1186);
//    CA_C_O_angle = (M_PI - 2.1033);
//    omega_const = -3.1318;
     //backbone angles [Engh & Huber angles]
    C_N_CA_angle = (M_PI - 2.1241);
    N_CA_C_angle = (M_PI - 2.0281);
    CA_C_N_angle = (M_PI - 1.9408);
    CA_C_O_angle = (M_PI - 2.0420);
    omega_const = -3.1318;

    //backbone distatnces
//    R_CA_C = 1.525;
//    R_C_N = 1.330;
//    R_N_CA = 1.460;

//backbone distatnces [Engh & Huber]
    R_CA_C = 1.525;
    R_C_N = 1.329;
    R_N_CA = 1.458;


    //C-beta
//    R_CA_CB = 1.52;
    // C_CA_CB_angle = -(M_PI - 1.9111);
//    C_CA_CB_angle = -(M_PI -M_PI*109.5/180.0);
//    N_C_CA_CB_diangle = 0.5*M_PI*122.6860/180.0;
//    correction_angle = 0.0;//-C_CA_CB_angle/2.0;

     //C-beta
    R_CA_CB = 1.53;
    // C_CA_CB_angle = -(M_PI - 1.9111);
    C_CA_CB_angle = -(M_PI -M_PI*110.1/180.0);
    N_C_CA_CB_diangle = 0.5*M_PI*122.6860/180.0;
    correction_angle = 0.0;//-C_CA_CB_angle/2.0;

//Original TPL values
    //OG serine
//    R_CB_OG = 1.417;
//    CA_CB_OG_angle = M_PI*110.773/180.0;
    //SG cysteine
//    R_CB_SG = 1.808;
//    CA_CB_SG_angle = M_PI*113.8169/180.0;
    //Valine CG1 and CG2
//    R_CB_CG = 1.527;
//    CA_CB_CG1_angle =  M_PI*110.7/180.0;
//    CG1_CB_CG2_angle = 1.88444687881;
    //Isoleucine
//    R_CG1_CD1 = 1.52;
//    CB_CG1_CD1_angle = M_PI*113.97/180.0;
    //Threonine
//    R_CB_OG1 = 1.43;
//    CA_CB_OG1_angle = M_PI*109.18/180.0;
//    OG1_CB_CG2_angle = 1.90291866134;
    //Arginine
//    R_C_C = R_CA_CB;
//    C_C_C_angle = C_CA_CB_angle;
//    R_CD_NE = 1.46;
//    R_NE_CZ = 1.33;
//    R_CZ_NH = 1.33;
//    CA_CB_CG_angle = -(M_PI - M_PI*113.83/180.0);
//    CB_CG_CD_angle = -(M_PI - M_PI*111.79/180.0);
//    CG_CD_NE_angle = -(M_PI - M_PI*111.68/180.0);
//    CD_NE_CZ_angle = -(M_PI - M_PI*124.79/180.0);
//    NE_CZ_NH_angle = M_PI*124.79/180.0;
    //Lysine
//    R_CD_CE = 1.46;
//    R_CE_NZ = 1.33;
//    CD_CE_NZ_angle = -(M_PI - M_PI*124.79/180.0);
    //Aspartic acid
//    R_CG_OD = 1.25;
//    CB_CG_OD_angle = M_PI*119.22/180.0;
//    OD1_CG_OD2_angle = 2.13911043783;
    //Asparagine
//    R_CG_OD1 = 1.23;
//    R_CG_ND2 = 1.33;
//    CB_CG_OD1_angle = M_PI*120.85/180.0;
//    CB_CG_ND2_angle = M_PI*116.48/180.0;
    //Methionine
//    R_CG_SD = 1.81;
//    R_SD_CE = 1.79;
//    CB_CG_SD_angle = -(M_PI - M_PI*112.69/180.0);
//    CG_SD_CE_angle = -(M_PI - M_PI*100.61/180.0);
//    //Histidine
//    R_CG_CD2 = 1.35;
//    R_CG_ND1 = 1.38;
//    R_CG_NE2 = 2.1912;
//    R_CG_CE1 = 2.1915;
//    CB_CG_CD2_angle = -(M_PI - 2.27957453603);
//    CB_CG_ND1_angle = (M_PI - 2.14413698608);
//    CB_CG_NE2_angle = -(M_PI - 2.90352974362);
//    CB_CG_CE1_angle = (M_PI - 2.75209584734);


// Engh & Huber
   //OG serine
    R_CB_OG = 1.417;
    CA_CB_OG_angle = M_PI*111.1/180.0;
    //SG cysteine
    R_CB_SG = 1.803;
    CA_CB_SG_angle = M_PI*114.4/180.0;
    //Valine CG1 and CG2
    R_CB_CG = 1.521;
    CA_CB_CG1_angle =  M_PI*110.5/180.0;
    CG1_CB_CG2_angle = 1.93382481121;
    //Isoleucine
    R_CG1_CD1 = 1.513;
    CB_CG1_CD1_angle = M_PI*113.8/180.0;
    //Threonine
    R_CB_OG1 = 1.433;
    CA_CB_OG1_angle = M_PI*109.6/180.0;
    OG1_CB_CG2_angle = 1.90764487243;
     //Arginine
    R_C_C = R_CA_CB;
    C_C_C_angle = C_CA_CB_angle;
    R_CD_NE = 1.460;
    R_NE_CZ = 1.329;
    R_CZ_NH = 1.326;
    CA_CB_CG_angle = -(M_PI - M_PI*111.2/180.0);
    CB_CG_CD_angle = -(M_PI - M_PI*111.3/180.0);
    CG_CD_NE_angle = -(M_PI - M_PI*112.0/180.0);
    CD_NE_CZ_angle = -(M_PI - M_PI*124.2/180.0);
    NE_CZ_NH_angle = M_PI*120.0/180.0;
    //Lysine
    R_CD_CE = 1.52;
    R_CE_NZ = 1.326;
    CD_CE_NZ_angle = -(M_PI - M_PI*111.9/180.0);
    //Aspartic acid
    R_CG_OD = 1.249;
    CB_CG_OD_angle = M_PI*118.4/180.0;
    OD1_CG_OD2_angle = 2.14500965070;
    //Asparagine
    R_CG_OD1 = 1.231;
    R_CG_ND2 = 1.328;
    CB_CG_OD1_angle = M_PI*120.85/180.0;
    CB_CG_ND2_angle = M_PI*116.4/180.0;
    //Methionine
    R_CG_SD = 1.803;
    R_SD_CE = 1.791;
    CB_CG_SD_angle = -(M_PI - M_PI*112.7/180.0);
    CG_SD_CE_angle = -(M_PI - M_PI*100.9/180.0);
    //Histidine
    R_CG_CD2 = 1.356;
    R_CG_ND1 = 1.371;
    R_CG_NE2 = 2.187;
    R_CG_CE1 = 2.167;//avg from 2.14289 + 2.19018
    CB_CG_CD2_angle = -(M_PI - 2.25322006432);
    CB_CG_ND1_angle = (M_PI - 2.12232037043);
    CB_CG_NE2_angle = -(M_PI - 2.89899188756);
    CB_CG_CE1_angle = (M_PI - 2.79078147394);


    //Geometry for Nucleic Acids
    //NA Backbone
    R_P_O5 = 1.593;
    R_O5_C5 = 1.440;
    R_C5_C4 = 1.511; // poly_type
    R_C4_C3 = 1.528; // poly_type
    R_C3_O3 = 1.431; // poly_type
    R_O3_P = 1.607;
    O3_P_O5_angle = (M_PI - (M_PI*104.0/180));
    P_O5_C5_angle = (M_PI - (M_PI*120.9/180));
    O5_C5_C4_angle = (M_PI - (M_PI*110.2/180)); // poly_type
    C5_C4_C3_angle = (M_PI - (M_PI*114.7/180)); // poly_type
    C4_C3_O3_angle = (M_PI - (M_PI*110.3/180)); // poly_type
    C3_O3_P_angle = (M_PI - (M_PI*119.7/180));

    //NA sugar ring
    R_C4_O4 = 1.446; // poly_type
    R_C3_C2 = 1.525; // poly_type
    R_C2_C1 = 1.528; // poly_type

    C3_C4_O4_angle = (M_PI - (M_PI*109.4/180)); //poly_type C5_C4_O4 109.4 || C3_C4_O4 105.6
    O3_C3_C2_angle = (M_PI - (M_PI*103.2/180)); //poly_type C4_C3_C2 103.2 || O3_C3_C2 110.6
    C3_C2_C1_angle = (M_PI - (M_PI*102.7/180)); //poly_type

    //C4_dummy_transform
    C4_correction_angle = 0.0;
    O3_C3_C4_O4_diangle = -0.5*((M_PI*252.20807/180)); //approximation from Gaussian

    //C3_dummy_transform
    C3_correction_angle = 0.0;
    P_O3_C3_C2_diangle = -0.5*(M_PI*93.58720/180); //approximation from Gaussian

//    DNA chi
    R_Cyt_N1 = 1.470;
    R_Gua_N1 = 1.459;
    R_Thy_N1 = 1.473;
    R_Ade_N1 = 1.462;
    R_Ura_N1 = 1.469;
    DNA_N1_angle = (M_PI - (M_PI * 114.2/180));

}

template <typename T> cGeometry<T>::~cGeometry(){
    
}

template <typename T> void cGeometry<T>::gly(){
    //backbone angles [Original TPL values]
//    C_N_CA_angle = (3.14159 - 1.9391);
//    N_CA_C_angle = (3.14159 - 2.061);
//    CA_C_N_angle = (3.14159 - 2.116);
//    CA_C_O_angle = (3.14159 - 2.1033);
//    omega_const = -3.1318;

//backbone distatnces
//    R_CA_C = 1.525;
//    R_C_N = 1.330;
//    R_N_CA = 1.460;


    //backbone angles [Engh & Huber angles]
    C_N_CA_angle = (M_PI - 2.1049);
    N_CA_C_angle = (M_PI - 1.9635);
    CA_C_N_angle = (M_PI - 2.0316);
    CA_C_O_angle = (M_PI - 2.0665);
    omega_const = -3.1318;
    //backbone distatnces
    R_CA_C = 1.516;
    R_C_N = 1.329;
    R_N_CA = 1.451;
}

template <typename T> void cGeometry<T>::ala(){
    //backbone angles [Original TPL values]
//    C_N_CA_angle = (3.14159 - 1.9391);
//    N_CA_C_angle = (3.14159 - 2.061);
//    CA_C_N_angle = (3.14159 - 2.1186);
//    CA_C_O_angle = (3.14159 - 2.1033);
//    omega_const = -3.1318;

    //backbone angles [Engh & Huber angles]
    C_N_CA_angle = (M_PI - 2.1241);
    N_CA_C_angle = (M_PI - 2.0281);
    CA_C_N_angle = (M_PI - 1.9408);
    CA_C_O_angle = (M_PI - 2.0420);
    omega_const = -3.1318;

    //backbone distatnces [Orig TPL]
//    R_CA_C = 1.525;
//    R_C_N = 1.330;
//    R_N_CA = 1.460;

    //backbone distatnces [E&H]
    R_CA_C = 1.525;
    R_C_N = 1.329;
    R_N_CA = 1.458;


    //C-beta [Original TPL values]
//    R_CA_CB = 1.52;
//    C_CA_CB_angle = -(3.14159 - 1.9111);//(3.14159 - 1.9111);
//    N_C_CA_CB_diangle = M_PI*111.0/180.0;;

    //C-beta [Engh and Huber Angles]
    R_CA_CB = 1.521;
    C_CA_CB_angle = -(3.14159 - 1.9286);//(3.14159 - 1.9111);
    N_C_CA_CB_diangle = M_PI*111.0/180.0;; // need dihedral angle
}

template class cGeometry<float>;
template class cGeometry<double>;