#include <torch/extension.h>
#include "coords2elec_interface.h"
#include "atomnames2params_interface.h"
#include "coords2stress_interface.h"
#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("AtomNames2Params_forward", &AtomNames2Params_forward, "Assigning parameters given atom names");
    m.def("AtomNames2Params_backward", &AtomNames2Params_backward, "Assigning parameters given atom names");
    
    m.def("Coords2Eps_forward", &Coords2Eps_forward, "Projecting coordinates to the grid dielectric constant");
    m.def("Coords2Eps_backward", &Coords2Eps_backward, "Projecting coordinates to the grid dielectric constant");
    m.def("Coords2Q_forward", &Coords2Q_forward, "Projecting coordinates to the grid charge");
    m.def("QEps2Phi_forward", &QEps2Phi_forward, "Solving electrostatics");

    m.def("Coords2Stress_forward", &Coords2Stress_forward, "Computing stress tensor from gamma");
    
}