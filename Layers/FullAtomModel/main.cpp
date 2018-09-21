#include <torch/torch.h>
#include <Angles2Coords/angles2coords_interface.h>
#include <PDB2Coords/pdb2coords_interface.h>
#include <Coords2TypedCoords/coords2typedcoords_interface.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("Angles2Coords_forward", &Angles2Coords_forward, "Angles2Coords forward");
	m.def("Angles2Coords_backward", &Angles2Coords_backward, "Angles2Coords backward");
	m.def("Angles2Coords_save", &Angles2Coords_save, "Angles2Coords save");
	m.def("getSeqNumAtoms", &getSeqNumAtoms, "Get number of atoms in a sequence");
	m.def("PDB2Coords", &PDB2Coords, "Convert PDB to coordinates");
	m.def("Coords2TypedCoords_forward", &Coords2TypedCoords_forward, "Convert coordinates to atom types");
	m.def("Coords2TypedCoords_backward", &Coords2TypedCoords_backward, "Backward of Coords2TypedCoords");
}
