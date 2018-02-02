#ifndef NUTIL_H_
#define NUTIL_H_
#include <string>
#include <algorithm>
#include <memory>

namespace StringUtil{
    //string utils
    std::string trim(const std::string &s);

    template<typename ... Args>
    inline std::string string_format( const std::string& format, Args ... args ){
        size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
        std::unique_ptr<char[]> buf( new char[ size ] ); 
        std::snprintf( buf.get(), size, format.c_str(), args ... );
        return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
    };
};

namespace ProtUtil{
    // atom indexing common for cConformation and cPDBLoader
    uint getAtomIndex(std::string &res_name, std::string &atom_name);
    
    // number of atoms in a sequence
    uint getNumAtoms(std::string &sequence, bool add_terminal);
    
    // heavy atoms
    bool isHeavyAtom(std::string &atom_name);

    // convert 1-letter aa code to 3-letter code
    std::string convertRes1to3(char resName);

    // assign atom type from 11 possible
    uint get11AtomType(std::string res_name, std::string atom_name, bool terminal);

};

#endif