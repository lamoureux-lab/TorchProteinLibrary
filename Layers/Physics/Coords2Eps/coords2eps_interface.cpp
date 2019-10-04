#include <torch/extension.h>
#include "coords2eps_interface.h"
#include <iostream>

torch::Tensor test(torch::Tensor coords, torch::Tensor num_atoms){
    torch::Tensor result = torch::zeros({30, 30, 30});
    auto coords_a = coords.accessor<double, 2>();
    auto num_atoms_a = num_atoms.accessor<int, 1>();
    std::cout<<num_atoms[0]<<std::endl;
    for(int i=0; i<num_atoms_a[0]; i++){
        // std::cout<<coords_a[0][3*i + 0]<<", ";
        // std::cout<<coords_a[0][3*i + 1]<<", ";
        // std::cout<<coords_a[0][3*i + 2]<<"\n";
    }
    return result;
}   