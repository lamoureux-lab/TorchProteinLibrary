#include <torch/extension.h>
#include <Coords2RMSD_CPU/coords2rmsd_interface.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("Coords2RMSD_CPU_forward", &Coords2RMSD_CPU_forward, "Coords2RMSD forward on cpu");
    m.def("Coords2RMSD_CPU_backward", &Coords2RMSD_CPU_backward, "Coords2RMSD backward on cpu");
}