#include <torch/extension.h>
#include "Coords2RMSD/coords2rmsd_interface.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("Coords2RMSDGPU_forward", &Coords2RMSDGPU_forward, "Coords2RMSD forward on GPU");
	m.def("Coords2RMSD_forward", &Coords2RMSD_forward, "Coords2RMSD forward on CPU");
}