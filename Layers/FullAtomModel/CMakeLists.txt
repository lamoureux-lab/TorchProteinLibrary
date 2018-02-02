ADD_LIBRARY(FULL_ATOM_MODEL nUtil.cpp cRigidGroup.cpp cConformation.cpp cConformationAA.cpp cGeometry.cpp cRMSD.cpp cPDBLoader.cpp)
TARGET_LINK_LIBRARIES(FULL_ATOM_MODEL TH_MATH TH)