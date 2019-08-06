#ifndef CRIGIDGROUP_H_
#define CRIGIDGROUP_H_
#include <cVector3.h>
#include <cMatrix44.h>
#include <vector>
#include <iostream>
#include <cGeometry.h>

template <typename T>
class cRigidGroup{
    public:
        std::vector<cVector3<T>> atoms_local;
        std::vector<cVector3<T>> atoms_global;
        std::vector<std::string> atomNames;
        std::vector<uint> atomIndexes;
        uint residueIndex;
        char residueName;
        cRigidGroup();
        ~cRigidGroup();
        void addAtom(cVector3<T> pos, std::string atomName, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
        void applyTransform(cMatrix44<T> mat);
};
template <typename T> cRigidGroup<T> *makeAtom(std::string atomName, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeCarbonyl(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr, bool terminal=false);
template <typename T> cRigidGroup<T> *makeSerGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeCysGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeValGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeIleGroup1(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeIleGroup2(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeLeuGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeThrGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeArgGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeAspGroup(cGeometry<T> &geo, std::string C, std::string O1, std::string O2, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeAsnGroup(cGeometry<T> &geo, std::string C, std::string O1, std::string N2, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeHisGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeProGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makePheGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeTyrGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> cRigidGroup<T> *makeTrpGroup(cGeometry<T> &geo, uint atomIndex, char residueName, uint residueIndex, T *atoms_global_ptr);
template <typename T> std::ostream& operator<<(std::ostream& os, const cRigidGroup<T>& rg);


#endif