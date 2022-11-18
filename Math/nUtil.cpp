#include <nUtil.h>
#include <iostream>
#include <stdarg.h>


//AS: Rotation and translation functions
template <typename T> void rotate(torch::Tensor &input_coords, cMatrix33<T> &R, torch::Tensor &output_coords, int num_atoms){
    T *data_in = input_coords.data<T>();
    T *data_out = output_coords.data<T>();
    for(int i=0;i<num_atoms;i++){
        cVector3<T> in(data_in+3*i);
        cVector3<T> out(data_out+3*i);
        out = R*in;
    }
};

template <typename T> void translate(torch::Tensor &input_coords, cVector3<T> &Tr, torch::Tensor &output_coords, int num_atoms){
    T *data_in = input_coords.data<T>();
    T *data_out = output_coords.data<T>();
    for(int i=0;i<num_atoms;i++){
        cVector3<T> in(data_in+3*i);
        cVector3<T> out(data_out+3*i);
        out = in + Tr;
    }
};

template <typename T> void computeBoundingBox(torch::Tensor &input_coords, int num_atoms, cVector3<T> &b0, cVector3<T> &b1){
    b0[0]=std::numeric_limits<T>::infinity(); 
    b0[1]=std::numeric_limits<T>::infinity(); 
    b0[2]=std::numeric_limits<T>::infinity();
    b1[0]=-1*std::numeric_limits<T>::infinity(); 
    b1[1]=-1*std::numeric_limits<T>::infinity(); 
    b1[2]=-1*std::numeric_limits<T>::infinity();
    T *data = input_coords.data<T>();
    for(int i=0;i<num_atoms;i++){
        cVector3<T> r(data + 3*i);
        if(r[0]<b0[0]){
            b0[0]=r[0];
        }
        if(r[1]<b0[1]){
            b0[1]=r[1];
        }
        if(r[2]<b0[2]){
            b0[2]=r[2];
        }

        if(r[0]>b1[0]){
            b1[0]=r[0];
        }
        if(r[1]>b1[1]){
            b1[1]=r[1];
        }
        if(r[2]>b1[2]){
            b1[2]=r[2];
        }
    }
};


template <typename T> cMatrix33<T> getRotation(T u1, T u2, T u3){
    T q[4];
    q[0] = sqrt(1-u1) * sin(2.0*M_PI*u2);
    q[1] = sqrt(1-u1) * cos(2.0*M_PI*u2);
    q[2] = sqrt(u1) * sin(2.0*M_PI*u3);
    q[3] = sqrt(u1) * cos(2.0*M_PI*u3);
    cMatrix33<T> rotation;
    rotation.m[0][0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
    rotation.m[0][1] = 2.0*(q[1]*q[2] - q[0]*q[3]);
    rotation.m[0][2] = 2.0*(q[1]*q[3] + q[0]*q[2]);

    rotation.m[1][0] = 2.0*(q[1]*q[2] + q[0]*q[3]);
    rotation.m[1][1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
    rotation.m[1][2] = 2.0*(q[2]*q[3] - q[0]*q[1]);

    rotation.m[2][0] = 2.0*(q[1]*q[3] - q[0]*q[2]);
    rotation.m[2][1] = 2.0*(q[2]*q[3] + q[0]*q[1]);
    rotation.m[2][2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
    
    return rotation;
};
template <typename T> cMatrix33<T> getRandomRotation(){
    torch::Tensor uni_rnd = torch::rand({3}, torch::TensorOptions().dtype(torch::kDouble));
    T u1 = uni_rnd.accessor<T,1>()[0];
    T u2 = uni_rnd.accessor<T,1>()[1];
    T u3 = uni_rnd.accessor<T,1>()[2];
    return getRotation(u1, u2, u3);
};
template <typename T> cVector3<T> getRandomTranslation(float spatial_dim, cVector3<T> &b0, cVector3<T> &b1){
    float dx_max = fmax(0, spatial_dim/2.0 - (b1[0]-b0[0])/2.0)*0.5;
    float dy_max = fmax(0, spatial_dim/2.0 - (b1[1]-b0[1])/2.0)*0.5;
    float dz_max = fmax(0, spatial_dim/2.0 - (b1[2]-b0[2])/2.0)*0.5;
    torch::Tensor uni_rnd = torch::rand({3}, torch::TensorOptions().dtype(torch::kDouble));
    
    uni_rnd[0] = uni_rnd[0]*(2.0*dx_max) - dx_max;
    uni_rnd[1] = uni_rnd[1]*(2.0*dy_max) - dy_max;
    uni_rnd[2] = uni_rnd[2]*(2.0*dz_max) - dz_max;
    auto acc = uni_rnd.accessor<T,1>();
    return cVector3<T>(acc[0], acc[1], acc[2]);
};


template <typename T> cMatrix33<T> tensor2Matrix33(torch::Tensor Ten){
    cMatrix33<T> dst;
    auto aT = Ten.accessor<T,2>();
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            dst.m[i][j] = aT[i][j];
    return dst;
};
template <typename T> void matrix2Tensor(cMatrix33<T> &mat, torch::Tensor &Ten){
    // auto aT = T.accessor<double,2>();
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            Ten[i][j] = mat.m[i][j];
};



std::string StringUtil::trim(const std::string &s){
   auto wsfront=std::find_if_not(s.begin(),s.end(),[](int c){return std::isspace(c);});
   auto wsback=std::find_if_not(s.rbegin(),s.rend(),[](int c){return std::isspace(c);}).base();
   return (wsback<=wsfront ? std::string() : std::string(wsfront,wsback));
}



std::string StringUtil::string_format(const std::string fmt, ...) {
    int size = ((int)fmt.size()) * 2 + 50;   // Use a rubric appropriate for your code
    std::string str;
    va_list ap;
    while (1) {     // Maximum two passes on a POSIX system...
        str.resize(size);
        va_start(ap, fmt);
        int n = vsnprintf((char *)str.data(), size, fmt.c_str(), ap);
        va_end(ap);
        if (n > -1 && n < size) {  // Everything worked
            str.resize(n);
            return str;
        }
        if (n > -1)  // Needed size returned
            size = n + 1;   // For null char
        else
            size *= 2;      // Guess at a larger size (OS specific)
    }
    return str;
}

torch::Tensor StringUtil::string2Tensor(std::string s){
    // torch::Tensor T = torch::CPU(torch::kByte).zeros({s.length()+1});
    torch::Tensor T = torch::zeros({s.length()+1}, torch::TensorOptions().dtype(torch::kByte));
    char* aT = static_cast<char*>(T.data_ptr());
    for(int i=0; i<s.length(); i++)
        aT[i] = s[i];
    aT[s.length()] = '\0';
    return T;
} 
void StringUtil::string2Tensor(std::string s, torch::Tensor T){
    char* aT = static_cast<char*>(T.data_ptr());
    for(int i=0; i<s.length(); i++)
        aT[i] = s[i];
    aT[s.length()] = '\0';
} 
std::string StringUtil::tensor2String(torch::Tensor T){
    std::string str;
    char* aT = static_cast<char*>(T.data_ptr());
    for(int i=0; i<T.size(0); i++){
        if(aT[i] == 0) break;
        str.push_back(aT[i]);
    }
    return str;
}
bool ProtUtil::isHeavyAtom(std::string &atom_name){
    if(atom_name[0] == 'C' || atom_name[0] == 'N' || atom_name[0] == 'O' || atom_name[0] == 'S' || atom_name[0] == 'P')
        return true;
    else
        return false;
}

bool ProtUtil::isNucleotide(std::string &res_name, int polymer_type){
    if(polymer_type == 1){
        if(res_name == "DA" || res_name == "DT" || res_name == "DC" || res_name == "DG")
            return true;
        else
            return false;
            }
    else if(polymer_type == 2){
        if(res_name == " A" || res_name == " U" || res_name == " C" || res_name == " G")
            return true;
        else
            return false;
            }
    else
        return false;
}

//AS: getNumAtoms returns num_atoms used in line 12 of simple_test.py?
uint ProtUtil::getNumAtoms(std::string &sequence, bool add_terminal){
    uint num_atoms = 0;
    std::string lastO("O");
    for(int i=0; i<sequence.length(); i++){
        std::string AA(1, sequence[i]);
        if(add_terminal){
            if( i<(sequence.length()-1) )
                lastO = "O";
            else
                lastO = "OXT";
        }else{
            lastO = "O";
        }
        num_atoms += getAtomIndex(AA, lastO) + 1;
    }
    return num_atoms;
}

uint ProtUtil::getAtomIndex(std::string &res_name, std::string &atom_name, bool fiveprime_ind, int polymer_type){

    if (polymer_type == 0){

    if(atom_name == std::string("N"))
        return 0;
    if(atom_name == std::string("CA"))
        return 1;
    if(atom_name == std::string("CB"))
        return 2;

    if(res_name == std::string("GLY") || res_name == std::string("G")){
        if(atom_name == std::string("C"))
            return 2;
        if(atom_name == std::string("O"))
            return 3;
        if(atom_name == std::string("OXT"))
            return 4;
    }

    if(res_name == std::string("ALA") || res_name == std::string("A")){
        if(atom_name == std::string("C"))
            return 3;
        if(atom_name == std::string("O"))
            return 4;
        if(atom_name == std::string("OXT"))
            return 5;
    }

    if(res_name == std::string("SER") || res_name == std::string("S")){
        if(atom_name == std::string("OG"))
            return 3;
        if(atom_name == std::string("C"))
            return 4;
        if(atom_name == std::string("O"))
            return 5;
        if(atom_name == std::string("OXT"))
            return 6;
    }

    if(res_name == std::string("CYS") || res_name == std::string("C")){
        if(atom_name == std::string("SG"))
            return 3;
        if(atom_name == std::string("C"))
            return 4;
        if(atom_name == std::string("O"))
            return 5;
        if(atom_name == std::string("OXT"))
            return 6;
    }

    if(res_name == std::string("VAL") || res_name == std::string("V")){
        if(atom_name == std::string("CG1"))
            return 3;
        if(atom_name == std::string("CG2"))
            return 4;
        if(atom_name == std::string("C"))
            return 5;
        if(atom_name == std::string("O"))
            return 6;
        if(atom_name == std::string("OXT"))
            return 7;
    }

    if(res_name == std::string("ILE") || res_name == std::string("I")){
        if(atom_name == std::string("CG2"))
            return 3;
        if(atom_name == std::string("CG1"))
            return 4;
        if(atom_name == std::string("CD1"))
            return 5;
        if(atom_name == std::string("C"))
            return 6;
        if(atom_name == std::string("O"))
            return 7;
        if(atom_name == std::string("OXT"))
            return 8;
    }

    if(res_name == std::string("LEU") || res_name == std::string("L")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD1"))
            return 4;
        if(atom_name == std::string("CD2"))
            return 5;
        if(atom_name == std::string("C"))
            return 6;
        if(atom_name == std::string("O"))
            return 7;
        if(atom_name == std::string("OXT"))
            return 8;
    }

    if(res_name == std::string("THR") || res_name == std::string("T")){
        if(atom_name == std::string("OG1"))
            return 3;
        if(atom_name == std::string("CG2"))
            return 4;
        if(atom_name == std::string("C"))
            return 5;
        if(atom_name == std::string("O"))
            return 6;
        if(atom_name == std::string("OXT"))
            return 7;
    }

    if(res_name == std::string("ARG") || res_name == std::string("R")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD"))
            return 4;
        if(atom_name == std::string("NE"))
            return 5;
        if(atom_name == std::string("CZ"))
            return 6;
        if(atom_name == std::string("NH1"))
            return 7;
        if(atom_name == std::string("NH2"))
            return 8;
        if(atom_name == std::string("C"))
            return 9;
        if(atom_name == std::string("O"))
            return 10;
        if(atom_name == std::string("OXT"))
            return 11;
    }

    if(res_name == std::string("LYS") || res_name == std::string("K")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD"))
            return 4;
        if(atom_name == std::string("CE"))
            return 5;
        if(atom_name == std::string("NZ"))
            return 6;
        if(atom_name == std::string("C"))
            return 7;
        if(atom_name == std::string("O"))
            return 8;
        if(atom_name == std::string("OXT"))
            return 9;
    }

    if(res_name == std::string("ASP") || res_name == std::string("D")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("OD1"))
            return 4;
        if(atom_name == std::string("OD2"))
            return 5;
        if(atom_name == std::string("C"))
            return 6;
        if(atom_name == std::string("O"))
            return 7;
        if(atom_name == std::string("OXT"))
            return 8;
    }

    if(res_name == std::string("ASN") || res_name == std::string("N")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("OD1"))
            return 4;
        if(atom_name == std::string("ND2"))
            return 5;
        if(atom_name == std::string("C"))
            return 6;
        if(atom_name == std::string("O"))
            return 7;
        if(atom_name == std::string("OXT"))
            return 8;
    }

    if(res_name == std::string("GLU") || res_name == std::string("E")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD"))
            return 4;
        if(atom_name == std::string("OE1"))
            return 5;
        if(atom_name == std::string("OE2"))
            return 6;
        if(atom_name == std::string("C"))
            return 7;
        if(atom_name == std::string("O"))
            return 8;
        if(atom_name == std::string("OXT"))
            return 9;
    }

    if(res_name == std::string("GLN") || res_name == std::string("Q")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD"))
            return 4;
        if(atom_name == std::string("OE1"))
            return 5;
        if(atom_name == std::string("NE2"))
            return 6;
        if(atom_name == std::string("C"))
            return 7;
        if(atom_name == std::string("O"))
            return 8;
        if(atom_name == std::string("OXT"))
            return 9;
    }

    if(res_name == std::string("MET") || res_name == std::string("M")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("SD"))
            return 4;
        if(atom_name == std::string("CE"))
            return 5;
        if(atom_name == std::string("C"))
            return 6;
        if(atom_name == std::string("O"))
            return 7;
        if(atom_name == std::string("OXT"))
            return 8;
    }

    if(res_name == std::string("HIS") || res_name == std::string("H")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD2"))
            return 4;
        if(atom_name == std::string("NE2"))
            return 5;
         if(atom_name == std::string("CE1"))
            return 6;
        if(atom_name == std::string("ND1"))
            return 7;
        if(atom_name == std::string("C"))
            return 8;
        if(atom_name == std::string("O"))
            return 9;
        if(atom_name == std::string("OXT"))
            return 10;
    }

    if(res_name == std::string("PRO") || res_name == std::string("P")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD"))
            return 4;
        if(atom_name == std::string("C"))
            return 5;
        if(atom_name == std::string("O"))
            return 6;
        if(atom_name == std::string("OXT"))
            return 7;
    }

    if(res_name == std::string("PHE") || res_name == std::string("F")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD1"))
            return 4;
        if(atom_name == std::string("CE1"))
            return 5;
         if(atom_name == std::string("CZ"))
            return 6;
        if(atom_name == std::string("CE2"))
            return 7;
        if(atom_name == std::string("CD2"))
            return 8;
        if(atom_name == std::string("C"))
            return 9;
        if(atom_name == std::string("O"))
            return 10;
        if(atom_name == std::string("OXT"))
            return 11;
    }

    if(res_name == std::string("TYR") || res_name == std::string("Y")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD1"))
            return 4;
        if(atom_name == std::string("CE1"))
            return 5;
         if(atom_name == std::string("CZ"))
            return 6;
        if(atom_name == std::string("CE2"))
            return 7;
        if(atom_name == std::string("CD2"))
            return 8;
        if(atom_name == std::string("OH"))
            return 9;
        if(atom_name == std::string("C"))
            return 10;
        if(atom_name == std::string("O"))
            return 11;
        if(atom_name == std::string("OXT"))
            return 12;
    }

    if(res_name == std::string("TRP") || res_name == std::string("W")){
        if(atom_name == std::string("CG"))
            return 3;
        if(atom_name == std::string("CD1"))
            return 4;
        if(atom_name == std::string("CD2"))
            return 5;
         if(atom_name == std::string("NE1"))
            return 6;
        if(atom_name == std::string("CE2"))
            return 7;
        if(atom_name == std::string("CE3"))
            return 8;
        if(atom_name == std::string("CZ2"))
            return 9;
        if(atom_name == std::string("CZ3"))
            return 10;
        if(atom_name == std::string("CH2"))
            return 11;
        if(atom_name == std::string("C"))
            return 12;
        if(atom_name == std::string("O"))
            return 13;
        if(atom_name == std::string("OXT"))
            return 14;
    }
    }

    if( polymer_type = 1){
    if (fiveprime_ind && res_name == std::string("DA")){
        if(atom_name == std::string("O5'"))
            return 0;
        if(atom_name == std::string("C5'"))
            return 1;
        if(atom_name == std::string("C4'"))
            return 2;
        if(atom_name == std::string("O4'"))
            return 3;
        if(atom_name == std::string("C3'"))
            return 4;
        if(atom_name == std::string("O3'"))
            return 5;
        if(atom_name == std::string("C2'"))
            return 6;
        if(atom_name == std::string("C1'"))
            return 7;
        if(atom_name == std::string("N9"))
            return 8;
        if(atom_name == std::string("C8"))
            return 9;
        if(atom_name == std::string("N7"))
            return 10;
        if(atom_name == std::string("C5"))
            return 11;
        if(atom_name == std::string("C6"))
            return 12;
        if(atom_name == std::string("N6"))
            return 13;
        if(atom_name == std::string("N1"))
            return 14;
        if(atom_name == std::string("C2"))
            return 15;
        if(atom_name == std::string("N3"))
            return 16;
        if(atom_name == std::string("C4"))
            return 17;
    }

     if(res_name == std::string("DG") && fiveprime_ind){
        if(atom_name == std::string("O5'"))
            return 0;
        if(atom_name == std::string("C5'"))
            return 1;
        if(atom_name == std::string("C4'"))
            return 2;
        if(atom_name == std::string("O4'"))
            return 3;
        if(atom_name == std::string("C3'"))
            return 4;
        if(atom_name == std::string("O3'"))
            return 5;
        if(atom_name == std::string("C2'"))
            return 6;
        if(atom_name == std::string("C1'"))
            return 7;
        if(atom_name == std::string("N9"))
            return 8;
        if(atom_name == std::string("C8"))
            return 9;
        if(atom_name == std::string("N7"))
            return 10;
        if(atom_name == std::string("C5"))
            return 11;
        if(atom_name == std::string("C6"))
            return 12;
        if(atom_name == std::string("O6"))
            return 13;
        if(atom_name == std::string("N1"))
            return 14;
        if(atom_name == std::string("C2"))
            return 15;
        if(atom_name == std::string("N2"))
            return 16;
        if(atom_name == std::string("N3"))
            return 17;
        if(atom_name == std::string("C4"))
            return 18;
    }

    if(res_name == std::string("DT") && fiveprime_ind){
        if(atom_name == std::string("O5'"))
            return 0;
        if(atom_name == std::string("C5'"))
            return 1;
        if(atom_name == std::string("C4'"))
            return 2;
        if(atom_name == std::string("O4'"))
            return 3;
        if(atom_name == std::string("C3'"))
            return 4;
        if(atom_name == std::string("O3'"))
            return 5;
        if(atom_name == std::string("C2'"))
            return 6;
        if(atom_name == std::string("C1'"))
            return 7;
        if(atom_name == std::string("N1"))
            return 8;
        if(atom_name == std::string("C2"))
            return 9;
        if(atom_name == std::string("O2"))
            return 10;
        if(atom_name == std::string("N3"))
            return 11;
        if(atom_name == std::string("C4"))
            return 12;
        if(atom_name == std::string("O4"))
            return 13;
        if(atom_name == std::string("C5"))
            return 14;
        if(atom_name == std::string("C7"))
            return 15;
        if(atom_name == std::string("C6"))
            return 16;
        }

    if(res_name == std::string("DC") && fiveprime_ind){
        if(atom_name == std::string("O5'"))
            return 0;
        if(atom_name == std::string("C5'"))
            return 1;
        if(atom_name == std::string("C4'"))
            return 2;
        if(atom_name == std::string("O4'"))
            return 3;
        if(atom_name == std::string("C3'"))
            return 4;
        if(atom_name == std::string("O3'"))
            return 5;
        if(atom_name == std::string("C2'"))
            return 6;
        if(atom_name == std::string("C1'"))
            return 7;
        if(atom_name == std::string("N1"))
            return 8;
        if(atom_name == std::string("C2"))
            return 9;
        if(atom_name == std::string("O2"))
            return 10;
        if(atom_name == std::string("N3"))
            return 11;
        if(atom_name == std::string("C4"))
            return 12;
        if(atom_name == std::string("N4"))
            return 13;
        if(atom_name == std::string("C5"))
            return 14;
        if(atom_name == std::string("C6"))
            return 15;
    }
    if((res_name == std::string("DA") && !fiveprime_ind ))}{//|| res_name == std::string("O")){
        if(atom_name == std::string("O5'"))
            return 3;
        if(atom_name == std::string("C5'"))
            return 4;
        if(atom_name == std::string("C4'"))
            return 5;
        if(atom_name == std::string("O4'"))
            return 6;
        if(atom_name == std::string("C3'"))
            return 7;
        if(atom_name == std::string("O3'"))
            return 8;
        if(atom_name == std::string("C2'"))
            return 9;
        if(atom_name == std::string("C1'"))
            return 10;
        if(atom_name == std::string("N9"))
            return 11;
        if(atom_name == std::string("C8"))
            return 12;
        if(atom_name == std::string("N7"))
            return 13;
        if(atom_name == std::string("C5"))
            return 14;
        if(atom_name == std::string("C6"))
            return 15;
        if(atom_name == std::string("N6"))
            return 16;
        if(atom_name == std::string("N1"))
            return 17;
        if(atom_name == std::string("C2"))
            return 18;
        if(atom_name == std::string("N3"))
            return 19;
        if(atom_name == std::string("C4"))
            return 20;
    }

     if(res_name == std::string("DG") && !fiveprime_ind){// || res_name == std::string("U")){
        if(atom_name == std::string("P"))
            return 0;
        if(atom_name == std::string("OP1"))
            return 1;
        if(atom_name == std::string("OP2"))
            return 2;
        if(atom_name == std::string("O5'"))
            return 3;
        if(atom_name == std::string("C5'"))
            return 4;
        if(atom_name == std::string("C4'"))
            return 5;
        if(atom_name == std::string("O4'"))
            return 6;
        if(atom_name == std::string("C3'"))
            return 7;
        if(atom_name == std::string("O3'"))
            return 8;
        if(atom_name == std::string("C2'"))
            return 9;
        if(atom_name == std::string("C1'"))
            return 10;
        if(atom_name == std::string("N9"))
            return 11;
        if(atom_name == std::string("C8"))
            return 12;
        if(atom_name == std::string("N7"))
            return 13;
        if(atom_name == std::string("C5"))
            return 14;
        if(atom_name == std::string("C6"))
            return 15;
        if(atom_name == std::string("O6"))
            return 16;
        if(atom_name == std::string("N1"))
            return 17;
        if(atom_name == std::string("C2"))
            return 18;
        if(atom_name == std::string("N2"))
            return 19;
        if(atom_name == std::string("N3"))
            return 20;
        if(atom_name == std::string("C4"))
            return 21;
    }

    if(res_name == std::string("DT") && !fiveprime_ind){// || res_name == std::string("Z")){
        if(atom_name == std::string("P"))
            return 0;
        if(atom_name == std::string("OP1"))
            return 1;
        if(atom_name == std::string("OP2"))
            return 2;
        if(atom_name == std::string("O5'"))
            return 3;
        if(atom_name == std::string("C5'"))
            return 4;
        if(atom_name == std::string("C4'"))
            return 5;
        if(atom_name == std::string("O4'"))
            return 6;
        if(atom_name == std::string("C3'"))
            return 7;
        if(atom_name == std::string("O3'"))
            return 8;
        if(atom_name == std::string("C2'"))
            return 9;
        if(atom_name == std::string("C1'"))
            return 10;
        if(atom_name == std::string("N1"))
            return 11;
        if(atom_name == std::string("C2"))
            return 12;
        if(atom_name == std::string("O2"))
            return 13;
        if(atom_name == std::string("N3"))
            return 14;
        if(atom_name == std::string("C4"))
            return 15;
        if(atom_name == std::string("O4"))
            return 16;
        if(atom_name == std::string("C5"))
            return 17;
        if(atom_name == std::string("C7"))
            return 18;
        if(atom_name == std::string("C6"))
            return 19;
        }

    if(res_name == std::string("DC") && !fiveprime_ind){ // || res_name == std::string("B")){
        if(atom_name == std::string("P"))
            return 0;
        if(atom_name == std::string("OP1"))
            return 1;
        if(atom_name == std::string("OP2"))
            return 2;
        if(atom_name == std::string("O5'"))
            return 3;
        if(atom_name == std::string("C5'"))
            return 4;
        if(atom_name == std::string("C4'"))
            return 5;
        if(atom_name == std::string("O4'"))
            return 6;
        if(atom_name == std::string("C3'"))
            return 7;
        if(atom_name == std::string("O3'"))
            return 8;
        if(atom_name == std::string("C2'"))
            return 9;
        if(atom_name == std::string("C1'"))
            return 10;
        if(atom_name == std::string("N1"))
            return 11;
        if(atom_name == std::string("C2"))
            return 12;
        if(atom_name == std::string("O2"))
            return 13;
        if(atom_name == std::string("N3"))
            return 14;
        if(atom_name == std::string("C4"))
            return 15;
        if(atom_name == std::string("N4"))
            return 16;
        if(atom_name == std::string("C5"))
            return 17;
        if(atom_name == std::string("C6"))
            return 18;
        }
        }

        if (polymer_type == 2){
        std::cout << "Get Atom Index not implemented for Polymer Type 2 \n";
        }

    std::cout<<"Unknown atom/res names"<<std::endl;
    throw(std::string("Unknown atom/res names"));
}

std::string ProtUtil::convertRes1to3(char resName){
    switch(resName){
        case 'G':
            return std::string("GLY");
        case 'A':
            return std::string("ALA");
        case 'S':
            return std::string("SER");
        case 'C':
            return std::string("CYS");
        case 'V':
            return std::string("VAL");
        case 'I':
            return std::string("ILE");
        case 'L':
            return std::string("LEU");   
        case 'T':
            return std::string("THR");    
        case 'R':
            return std::string("ARG");
        case 'K':
            return std::string("LYS");
        case 'D':
            return std::string("ASP");
        case 'N':
            return std::string("ASN");
        case 'E':
            return std::string("GLU");
        case 'Q':
            return std::string("GLN");
        case 'M':
            return std::string("MET");
        case 'H':
            return std::string("HIS");
        case 'P':
            return std::string("PRO");
        case 'F':
            return std::string("PHE");
        case 'Y':
            return std::string("TYR");
        case 'W':
            return std::string("TRP");
        default:
            std::cout<<"Unknown residue name"<<std::endl;
            throw("Unknown residue name");
    }
}

uint ProtUtil::get4AtomTypeElement(std::string res_name, std::string atom_name, bool terminal){
	auto f = [](unsigned char const c) { return std::isspace(c); };
	atom_name.erase(std::remove_if(atom_name.begin(), atom_name.end(), f), atom_name.end());
	uint assignedType = 0;
	std::string fullAtomName;

	if(atom_name[0] == 'C'){
	  assignedType = 0;
	}else if(atom_name[0] == 'N'){
	  assignedType = 1;
	}else if(atom_name[0] == 'O'){
	  assignedType = 2;
	}else if(atom_name[0] == 'S'){
	  assignedType = 3;
	}else{
	  throw std::string("Unknown atom type") + res_name + atom_name;
	}
	return assignedType;
}


uint ProtUtil::get11AtomType(std::string res_name, std::string atom_name, bool terminal){
	auto f = [](unsigned char const c) { return std::isspace(c); };
	atom_name.erase(std::remove_if(atom_name.begin(), atom_name.end(), f), atom_name.end());
	uint assignedType = 0;
	std::string fullAtomName;

	// dealing with the residue-agnostic atom types
	if(atom_name==std::string("O")){
		if(terminal)
			assignedType = 8;
		else
			assignedType = 6;
	}else if(atom_name==std::string("OXT")){
			assignedType = 8;
	}else if(atom_name==std::string("OT2")){
			assignedType = 8;
	}else if(atom_name==std::string("N")){
		assignedType = 2;
	}else if(atom_name==std::string("C")){
		assignedType = 9;
	}else if(atom_name==std::string("CA")){
		assignedType = 11;
	}else{
		// dealing with the residue-dependent atom types
		fullAtomName = res_name + atom_name;
		
		if(fullAtomName == std::string("CYSSG") || fullAtomName == std::string("METSD") || fullAtomName == std::string("MSESE")){
			assignedType = 1;
		}else
		// 2 amide N (original ITScore groups 9 + 10)
		if(fullAtomName == std::string("ASNND2") || fullAtomName == std::string("GLNNE2")){
			assignedType = 2;
		}else
		// 3  Nar Aromatic nitrogens (original ITScore group 11)
		if(fullAtomName == std::string("HISND1") || fullAtomName == std::string("HISNE2") || fullAtomName == std::string("TRPNE1")){
			assignedType = 3;
		}else
		// 4 guanidine N (original ITScore groups 12 +13 )
		if(fullAtomName == std::string("ARGNH1") || fullAtomName == std::string("ARGNH2") || fullAtomName == std::string("ARGNE")){
			assignedType = 4;
		}else
		// 5 N31 Nitrogen with three hydrogens (original ITScore 14)
		if(fullAtomName == std::string("LYSNZ")){
			assignedType = 5;
		}else
		// 6 carboxyl 0 (original ITScore groups 15+16)
		if(fullAtomName == std::string("ACEO") || fullAtomName == std::string("ASNOD1") || fullAtomName == std::string("GLNOE1")){
			assignedType = 6;
		}else
		// 7 O3H Oxygens in hydroxyl groups (original ITScore group 17)
		if(fullAtomName == std::string("SEROG") || fullAtomName == std::string("THROG1") || fullAtomName == std::string("TYROH")){
			assignedType = 7;
		}else
		// 8 O22 Oxygens in carboxyl groups, terminus oxygens (original ITScore group 18)
		if(fullAtomName == std::string("ASPOD1") || fullAtomName == std::string("ASPOD2") || \
			fullAtomName == std::string("GLUOE1") || fullAtomName == std::string("GLUOE2")){
			assignedType = 8;
		}else
		// 9 Sp2 Carbons (groups ITScore 1 + 2 + 3 + 4)
		if(fullAtomName == std::string("ARGCZ") || fullAtomName == std::string("ASPCG") || \
			fullAtomName == std::string("GLUCD") || fullAtomName == std::string("ACEC") || \
			fullAtomName == std::string("ASNCG") || fullAtomName == std::string("GLNCD")){
			assignedType = 9;
		}else
		// 10 Car Aromatic carbons (groups ITScore 5)
		if(fullAtomName == std::string("HISCD2") || fullAtomName == std::string("HISCE1") || \
			fullAtomName == std::string("HISCG") || \
			\
			fullAtomName == std::string("PHECD1") || fullAtomName == std::string("PHECD2") || \
			fullAtomName == std::string("PHECE1") || fullAtomName == std::string("PHECE2")|| \
			fullAtomName == std::string("PHECG") || fullAtomName == std::string("PHECZ") || \
			\
			fullAtomName == std::string("TRPCD1") || fullAtomName == std::string("TRPCD2") || \
			fullAtomName == std::string("TRPCE2") || fullAtomName == std::string("TRPCE3") || \
			fullAtomName == std::string("TRPCG") || fullAtomName == std::string("TRPCH2") || \
			fullAtomName == std::string("TRPCZ2") || fullAtomName == std::string("TRPCZ3") || \
			\
			fullAtomName == std::string("TYRCD1") || fullAtomName == std::string("TYRCD2") || \
			fullAtomName == std::string("TYRCE1") || fullAtomName == std::string("TYRCE2") || \
			fullAtomName == std::string("TYRCG") || fullAtomName == std::string("TYRCZ")){
			assignedType = 10;
		}else
		// 11 Sp3 Carbons (corresponding ITScore groups 6 + 7 + 8)
		if(fullAtomName == std::string("ALACB") || \
			\
			fullAtomName == std::string("ARGCB") || fullAtomName == std::string("ARGCG") || fullAtomName == std::string("ARGCD") || \
			\
			fullAtomName == std::string("ASNCB") || \
			\
			fullAtomName == std::string("ASPCB") || \
			\
			fullAtomName == std::string("GLNCB") || fullAtomName == std::string("GLNCG") || \
			\
			fullAtomName == std::string("GLUCB") || fullAtomName == std::string("GLUCG")|| \
			\
			fullAtomName == std::string("HISCB") || \
			\
			fullAtomName == std::string("ILECB") || fullAtomName == std::string("ILECD1") || \
			fullAtomName == std::string("ILECG1") || fullAtomName == std::string("ILECG2") || \
			\
			fullAtomName == std::string("LEUCB") || fullAtomName == std::string("LEUCD1") || \
			fullAtomName == std::string("LEUCD2") || fullAtomName == std::string("LEUCG") || \
			\
			fullAtomName == std::string("LYSCB") || fullAtomName == std::string("LYSCD") || \
			fullAtomName == std::string("LYSCG") || fullAtomName == std::string("LYSCE") || \
			\
			fullAtomName == std::string("METCB") || fullAtomName == std::string("METCE") || fullAtomName == std::string("METCG") || \
			\
			fullAtomName == std::string("MSECB") || fullAtomName == std::string("MSECE") || fullAtomName == std::string("MSECG") || \
			\
			fullAtomName == std::string("PHECB") || \
			\
			fullAtomName == std::string("PROCB") || fullAtomName == std::string("PROCG") || fullAtomName == std::string("PROCD") || \
			\
			fullAtomName == std::string("SERCB") || \
			\
			fullAtomName == std::string("THRCG2") || \
			\
			fullAtomName == std::string("TRPCB") || \
			\ 
			fullAtomName == std::string("TYRCB") || \
			\
			fullAtomName == std::string("VALCB") || fullAtomName == std::string("VALCG1") || fullAtomName == std::string("VALCG2") || \
			\
			fullAtomName == std::string("ACECH3") || \
			\
			fullAtomName == std::string("THRCB") || \
			\
			fullAtomName == std::string("CYSCB") ){
			assignedType = 11;
		}else{
            // std::cout<<std::string("Unknown atom type") + res_name + atom_name<<std::endl;
			throw std::string("Unknown atom type") + res_name + atom_name;
		}

	}
	return assignedType - 1;
}

uint ProtUtil::getAtomTypeCharmm(std::string res_name, std::string atom_name, bool terminal){
	auto f = [](unsigned char const c) { return std::isspace(c); };
	atom_name.erase(std::remove_if(atom_name.begin(), atom_name.end(), f), atom_name.end());
	uint assignedType = 0;
	std::string fullAtomName;

	// dealing with backbone & CB atom types
	if(atom_name==std::string("C")){
	  assignedType = 0;
	}else if(atom_name==std::string("CA")){
	  if(res_name==std::string("GLY")){
	    assignedType = 3;
	  }else if(res_name==std::string("PRO")){
	    assignedType = 10;
	  }else{
	    assignedType = 2;
	  }
	}else if(atom_name==std::string("CB")){
	  if(res_name==std::string("ILE") || res_name==std::string("THR") || res_name==std::string("VAL")){
	    assignedType = 2;
	  }else if(res_name==std::string("ASP") || res_name==std::string("GLU")){
	    assignedType = 4;
	  }else if(res_name==std::string("ALA")){
	    assignedType = 5;
	  }else if(res_name==std::string("PRO")){
	    assignedType = 11;
	  }else{
	    assignedType = 3;
	  }
	}else if(atom_name==std::string("N")){
	  if(res_name==std::string("PRO")){
	    assignedType = 15;
	  }else{
	    assignedType = 17;
	  }
	}else if(atom_name==std::string("O")){
	  assignedType = 22;
	}else if(atom_name==std::string("OXT") || atom_name==std::string("OT2")){
	  assignedType = 23;
	}else{

	  // dealing with the residue-dependent atom types
	  fullAtomName = res_name + atom_name;
	  
	  if(fullAtomName == std::string("ARGCZ")){
	    assignedType = 0;
	  }else if(fullAtomName == std::string("PHECG") || fullAtomName == std::string("PHECD1") || \
		   fullAtomName == std::string("PHECD2") || fullAtomName == std::string("PHECE1") || \
		   fullAtomName == std::string("PHECE2") || fullAtomName == std::string("PHECZ") || \
		   fullAtomName == std::string("TRPCD1") || fullAtomName == std::string("TRPCZ3") || \
		   fullAtomName == std::string("TYRCG") || fullAtomName == std::string("TYRCD1") || \
		   fullAtomName == std::string("TYRCE1") || fullAtomName == std::string("TYRCZ") || \
		   fullAtomName == std::string("TYRCE2") || fullAtomName == std::string("TYRCD2") || \
		   fullAtomName == std::string("TRPCH2")){
	    assignedType = 1;
	  }else if(fullAtomName == std::string("LEUCG")){
	    assignedType = 2;
	  }else if(fullAtomName == std::string("ARGCG") || fullAtomName == std::string("ARGCD") || \
		   fullAtomName == std::string("GLNCG") || fullAtomName == std::string("ILECG1") || \	
		   fullAtomName == std::string("GLUCG") || fullAtomName == std::string("LYSCG") || \
		   fullAtomName == std::string("LYSCD") || fullAtomName == std::string("LYSCE") || \
		   fullAtomName == std::string("METCG")){ 
	    assignedType = 3;
	  }else if(fullAtomName == std::string("ILECG2") || fullAtomName == std::string("VALCG2") || \
		   fullAtomName == std::string("ILECD1") || fullAtomName == std::string("LEUCD1") || \
		   fullAtomName == std::string("LEUCD2") || fullAtomName == std::string("METCE") || \
		   fullAtomName == std::string("THRCG2") || fullAtomName == std::string("VALCG1")){  
	    assignedType = 5;
	  }else if(fullAtomName == std::string("HISCG") || fullAtomName == std::string("HISCD2")){
	    assignedType = 6;
	  }else if(fullAtomName == std::string("HISCE1")){
	    assignedType = 7;
	  }else if(fullAtomName == std::string("TRPCD2") || fullAtomName == std::string("TRPCE2")){
	    assignedType = 8;
	  }else if(fullAtomName == std::string("TRPCG")){
	    assignedType = 9;
	  }else if(fullAtomName == std::string("PROCG")){
	    assignedType = 11;
	  }else if(fullAtomName == std::string("PROCD")){
	    assignedType = 12;
	  }else if(fullAtomName == std::string("ASNCG") || fullAtomName == std::string("ASPCG") || \
		   fullAtomName == std::string("GLNCD") || fullAtomName == std::string("GLUCD")){
	    assignedType = 13;
	  }else if(fullAtomName == std::string("TRPCE3") || fullAtomName == std::string("TRPCZ2")){
	    assignedType = 14;
	  }else if(fullAtomName == std::string("HISND1") || fullAtomName == std::string("HISNE2")){
	    assignedType = 16;
	  }else if(fullAtomName == std::string("ASNND2") || fullAtomName == std::string("GLNNE2")){
	    assignedType = 18;
	  }else if(fullAtomName == std::string("LYSNZ")){
	    assignedType = 19;
	  }else if(fullAtomName == std::string("ARGNE") || fullAtomName == std::string("ARGNH1") || \
		   fullAtomName == std::string("ARGNH2") ){
	    assignedType = 20;
	  }else if(fullAtomName == std::string("TRPNE1")){
	    assignedType = 21;
	  }else if(fullAtomName == std::string("ASNOD1") || fullAtomName == std::string("GLNOE1")){
	    assignedType = 22;
	  }else if(fullAtomName == std::string("ASPOD1") || fullAtomName == std::string("ASPOD2") || \
		   fullAtomName == std::string("GLUOE1") || fullAtomName == std::string("GLUOE2")){
	    assignedType = 23;
	  }else if(fullAtomName == std::string("SEROG") || fullAtomName == std::string("THROG1") ||  fullAtomName == std::string("TYROH")){
	    assignedType = 24;
	  }else if(fullAtomName == std::string("CYSSG") || fullAtomName == std::string("METSD")){
	    assignedType = 25;
	  }else{
	    throw std::string("Unknown atom type") + res_name + atom_name;
	  }

	}
	return assignedType ;
}

	  


template void rotate(torch::Tensor&, cMatrix33<float>&, torch::Tensor&, int);
template void translate(torch::Tensor&, cVector3<float>&, torch::Tensor&, int);
template void computeBoundingBox(torch::Tensor&, int, cVector3<float>&, cVector3<float>&);
template cMatrix33<float> getRotation(float u1, float u2, float u3);
template cMatrix33<float> getRandomRotation();
template cVector3<float> getRandomTranslation(float spatial_dim, cVector3<float>&, cVector3<float>&);
template cMatrix33<float> tensor2Matrix33(torch::Tensor);
template void matrix2Tensor(cMatrix33<float>&, torch::Tensor&);

template void rotate(torch::Tensor&, cMatrix33<double>&, torch::Tensor&, int);
template void translate(torch::Tensor&, cVector3<double>&, torch::Tensor&, int);
template void computeBoundingBox(torch::Tensor&, int, cVector3<double>&, cVector3<double>&);
template cMatrix33<double> getRotation(double u1, double u2, double u3);
template cMatrix33<double> getRandomRotation();
template cVector3<double> getRandomTranslation(float spatial_dim, cVector3<double>&, cVector3<double>&);
template cMatrix33<double> tensor2Matrix33(torch::Tensor);
template void matrix2Tensor(cMatrix33<double>&, torch::Tensor&);
