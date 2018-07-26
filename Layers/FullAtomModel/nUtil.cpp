#include <nUtil.h>
#include <iostream>
#include <stdarg.h>

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

void StringUtil::string2Tensor(std::string s, THByteTensor *T){
    if( (s.length()+1) > T->size[0])
        throw(std::string("String is longer than tensor"));

    for(int i=0; i<s.length(); i++)
        *((char*)THByteTensor_data(T) + i) = s[i];

    *((char*)THByteTensor_data(T) + s.length()) = '\0';
} 
std::string StringUtil::tensor2String(THByteTensor *T){
    return std::string( (const char*)THByteTensor_data(T));
}
bool ProtUtil::isHeavyAtom(std::string &atom_name){
    if(atom_name[0] == 'C' || atom_name[0] == 'N' || atom_name[0] == 'O' || atom_name[0] == 'S')
        return true;
    else
        return false;
}
cMatrix33 ProtUtil::tensor2Matrix33(THDoubleTensor *T){
    cMatrix33 dst;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            dst.m[i][j] = THDoubleTensor_get2d(T, i, j);
    return dst;
}
void ProtUtil::matrix2Tensor(cMatrix33 &mat, THDoubleTensor *T){
    for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                THDoubleTensor_set2d(T, i, j, mat.m[i][j]);
}
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
void ProtUtil::translate(THDoubleTensor *input_coords, cVector3 T, THDoubleTensor *output_coords, int num_atoms){
    // uint num_atoms = input_coords->size[0]/3;
    double *data_in = THDoubleTensor_data(input_coords);
    double *data_out = THDoubleTensor_data(output_coords);
    for(int i=0;i<num_atoms;i++){
        cVector3 in(data_in+3*i);
        cVector3 out(data_out+3*i);
        out = in + T;
    }
}
void ProtUtil::translate(THDoubleTensor *coords, cVector3 T){
    uint num_atoms = coords->size[0]/3;
    double *data = THDoubleTensor_data(coords);
    for(int i=0;i<num_atoms;i++){
        cVector3 r(data+3*i);
        r = r + T;
    }
}
void ProtUtil::rotate(THDoubleTensor *input_coords, cMatrix33 R, THDoubleTensor *output_coords, int num_atoms){
    double *data_in = THDoubleTensor_data(input_coords);
    double *data_out = THDoubleTensor_data(output_coords);
    for(int i=0;i<num_atoms;i++){
        cVector3 in(data_in+3*i);
        cVector3 out(data_out+3*i);
        out = R*in;
    }
}
void ProtUtil::rotate(THDoubleTensor *coords, cMatrix33 R){
    uint num_atoms = coords->size[0]/3;
    double *data = THDoubleTensor_data(coords);
    for(int i=0;i<num_atoms;i++){
        cVector3 r(data+3*i);
        r = R*r;
    }
}

cMatrix33 ProtUtil::getRandomRotation(THGenerator *gen){
    float u1 = THRandom_uniform(gen,0,1.0);
    float u2 = THRandom_uniform(gen,0,1.0);
    float u3 = THRandom_uniform(gen,0,1.0);
    float q[4];
    q[0] = sqrt(1-u1) * sin(2.0*M_PI*u2);
    q[1] = sqrt(1-u1) * cos(2.0*M_PI*u2);
    q[2] = sqrt(u1) * sin(2.0*M_PI*u3);
    q[3] = sqrt(u1) * cos(2.0*M_PI*u3);
    cMatrix33 random_rotation;
    random_rotation.m[0][0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
    random_rotation.m[0][1] = 2.0*(q[1]*q[2] - q[0]*q[3]);
    random_rotation.m[0][2] = 2.0*(q[1]*q[3] + q[0]*q[2]);

    random_rotation.m[1][0] = 2.0*(q[1]*q[2] + q[0]*q[3]);
    random_rotation.m[1][1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
    random_rotation.m[1][2] = 2.0*(q[2]*q[3] - q[0]*q[1]);

    random_rotation.m[2][0] = 2.0*(q[1]*q[3] - q[0]*q[2]);
    random_rotation.m[2][1] = 2.0*(q[2]*q[3] + q[0]*q[1]);
    random_rotation.m[2][2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
    
    return random_rotation;
}
cVector3 ProtUtil::getRandomTranslation(THGenerator *gen, uint spatial_dim, cVector3 b0, cVector3 b1){
    float dx_max = fmax(0, spatial_dim/2.0 - (b1[0]-b0[0])/2.0)*0.5;
    float dy_max = fmax(0, spatial_dim/2.0 - (b1[1]-b0[1])/2.0)*0.5;
    float dz_max = fmax(0, spatial_dim/2.0 - (b1[2]-b0[2])/2.0)*0.5;
    float dx = THRandom_uniform(gen,-dx_max,dx_max);
    float dy = THRandom_uniform(gen,-dy_max,dy_max);
    float dz = THRandom_uniform(gen,-dz_max,dz_max);
    return cVector3(dx, dy, dz);
}

void ProtUtil::computeBoundingBox(THDoubleTensor *input_coords, int num_atoms, cVector3 &b0, cVector3 &b1){
	b0[0]=std::numeric_limits<double>::infinity(); 
	b0[1]=std::numeric_limits<double>::infinity(); 
	b0[2]=std::numeric_limits<double>::infinity();
	b1[0]=-1*std::numeric_limits<double>::infinity(); 
	b1[1]=-1*std::numeric_limits<double>::infinity(); 
	b1[2]=-1*std::numeric_limits<double>::infinity();
    double *data = THDoubleTensor_data(input_coords);
    for(int i=0;i<num_atoms;i++){
        cVector3 r(data + 3*i);
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
}

uint ProtUtil::getAtomIndex(std::string &res_name, std::string &atom_name){
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
            return 5;
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
            std::cout<<std::string("Unknown atom type") + res_name + atom_name<<std::endl;
			throw std::string("Unknown atom type") + res_name + atom_name;
		}

	}
	return assignedType - 1;
}
