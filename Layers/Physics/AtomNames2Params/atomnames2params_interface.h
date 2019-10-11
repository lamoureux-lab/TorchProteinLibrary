#include <torch/extension.h>
#include <map>
#include <string>
#include <utility>

typedef std::pair<std::string, std::string> atom_type;
typedef std::pair<double, double> atom_param;
typedef std::pair<int, atom_param> indexes_atom_param;
typedef std::vector<std::pair<int, int>> assignment_indexes;

std::map<int, assignment_indexes> AtomNames2Params_forward(   torch::Tensor resnames, torch::Tensor atomnames, torch::Tensor num_atoms, 
                                                                    torch::Tensor types, torch::Tensor params, torch::Tensor assigned_types);

void AtomNames2Params_backward( torch::Tensor gradOutput, torch::Tensor gradInput, 
                                std::map<int, assignment_indexes> &indexes);