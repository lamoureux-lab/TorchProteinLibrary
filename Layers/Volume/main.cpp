#include <torch/extension.h>
#include <TypedCoords2Volume/typedcoords2volume_interface.h>
#include <Volume2Xplor/volume2xplor_interface.h>
#include <Select/select_interface.h>
#include <VolumeRMSD/volumeRMSD_interface.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("TypedCoords2Volume_forward", &TypedCoords2Volume_forward, "TypedCoords2Volume forward");
    m.def("TypedCoords2Volume_backward", &TypedCoords2Volume_backward, "TypedCoords2Volume backward");
    m.def("Volume2Xplor", &Volume2Xplor, "Save 3D volume as xplor file");
    m.def("SelectVolume_forward", &SelectVolume_forward, "Select feature columns from volume at coordinates");
    m.def("SelectVolume_backward", &SelectVolume_backward, "Backward of select feature columns from volume at coordinates");
    m.def("VolumeGenRMSD", &VolumeGenRMSD, "Generate RMSD on the circular grid of displacements");
}
