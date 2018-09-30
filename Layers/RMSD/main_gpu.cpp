#include <torch/torch.h>
#include <Coords2RMSD_GPU/coords2rmsd_interface.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("Coords2RMSD_GPU_forward", &Coords2RMSD_GPU_forward, "Coords2RMSD forward on gpu");
}