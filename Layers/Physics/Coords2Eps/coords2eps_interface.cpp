#include <torch/extension.h>
#include "coords2eps_interface.h"
#include <iostream>
#include <math.h>

torch::Tensor test(torch::Tensor coords, torch::Tensor num_atoms){
    torch::Tensor result = torch::zeros({30, 30, 30});
    AT_DISPATCH_FLOATING_TYPES(coords.type(), "tmp", ([&] {
        auto coords_a = coords.accessor<scalar_t, 2>();
        auto num_atoms_a = num_atoms.accessor<int, 1>();
        auto result_a = result.accessor<float, 3>();
        std::cout<<num_atoms[0]<<std::endl;
        
        scalar_t resolution = 2.5;
        for(int i=0; i<num_atoms_a[0]; i++){
            scalar_t x = coords_a[0][3*i + 0];
            scalar_t y = coords_a[0][3*i + 1];
            scalar_t z = coords_a[0][3*i + 2];
            int ix = floor(x/resolution);
            int iy = floor(y/resolution);
            int iz = floor(z/resolution);
            if((ix<0)||(ix>=30)||(iy<0)||(iy>=30)||(iz<0)||(iz>=30))
                continue;

            result_a[ix][iy][iz] = 1.0;
        }
    }));
    return result;
}   