if(NOT FAM_ONLY)
    CUDA_ADD_LIBRARY(FULL_ATOM_MODEL_CU STATIC Kernels.cu)
    TARGET_LINK_LIBRARIES(FULL_ATOM_MODEL_CU)
    ADD_LIBRARY(FULL_ATOM_MODEL STATIC nUtil.cpp cRigidGroup.cpp cConformation.cpp cConformationAA.cpp cGeometry.cpp cRMSD.cpp cPDBLoader.cpp)
    TARGET_LINK_LIBRARIES(FULL_ATOM_MODEL TH_MATH TH)
else(NOT FAM_ONLY)
    ADD_LIBRARY(FULL_ATOM_MODEL STATIC nUtil.cpp cRigidGroup.cpp cConformation.cpp cConformationAA.cpp cGeometry.cpp cRMSD.cpp cPDBLoader.cpp)
    TARGET_LINK_LIBRARIES(FULL_ATOM_MODEL TH_MATH TH)
endif(NOT FAM_ONLY)


