#include <torch/torch.h>
#include <Angles2Coords/angles2coords_interface.h>
#include <PDB2Coords/pdb2coords_interface.h>
#include <Coords2TypedCoords/coords2typedcoords_interface.h>
#include <CoordsTransform/coordsTransform_interface.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("Angles2Coords_forward", &Angles2Coords_forward, "Angles2Coords forward");
	m.def("Angles2Coords_backward", &Angles2Coords_backward, "Angles2Coords backward");
	m.def("Angles2Coords_save", &Angles2Coords_save, "Angles2Coords save");
	m.def("getSeqNumAtoms", &getSeqNumAtoms, "Get number of atoms in a sequence");
	m.def("PDB2CoordsOrdered", &PDB2CoordsOrdered, "Convert PDB to coordinates in the order of Angles2Coords output");
	m.def("PDB2CoordsUnordered", &PDB2CoordsUnordered, "Convert PDB to coordinates in the PDB order");
	m.def("Coords2TypedCoords_forward", &Coords2TypedCoords_forward, "Convert coordinates to atom types");
	m.def("Coords2TypedCoords_backward", &Coords2TypedCoords_backward, "Backward of Coords2TypedCoords");
	m.def("CoordsTranslate_forward", &CoordsTranslate_forward, "Translate coordinates");
	m.def("CoordsRotate_forward", &CoordsRotate_forward, "Rotate coordinates");
	m.def("CoordsRotate_backward", &CoordsRotate_backward, "Backward of rotate coordinates");
	m.def("getBBox", &getBBox, "Get bounding box of coordinates");
	m.def("getRandomRotation", &getRandomRotation, "Get random rotation");
	m.def("getRandomTranslation", &getRandomTranslation, "Get random translation");
}
