#include <torch/torch.h>
#include <TypedCoords2Volume/typedcoords2volume_interface.h>
#include <Volume2Xplor/volume2xplor_interface.h>
#include <Select/select_interface.h>
#include <VolumeConvolution/volumeConvolution_interface.h>
#include <VolumeRotation/volumeRotation_interface.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("TypedCoords2Volume_forward", &TypedCoords2Volume_forward, "TypedCoords2Volume forward");
    m.def("TypedCoords2Volume_backward", &TypedCoords2Volume_backward, "TypedCoords2Volume backward");
    m.def("Volume2Xplor", &Volume2Xplor, "Save 3D volume as xplor file");
    m.def("SelectVolume_forward", &SelectVolume_forward, "Select feature columns from volume at coordinates");
    m.def("VolumeConvolution_forward", &VolumeConvolution_forward, "VolumeConvolution forward");
    m.def("VolumeConvolution_backward", &VolumeConvolution_backward, "VolumeConvolution backward");
    m.def("VolumeGenGrid", &VolumeGenGrid, "Volume generate rotated grid");
}
