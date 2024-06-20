#include <torch/extension.h>
#include <Angles2Backbone/angles2backbone_interface.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("Angles2BackboneCPU_forward", &Angles2BackboneCPU_forward, "Angles2Backbone cpu forward");
    m.def("Angles2BackboneCPUAngles_backward", &Angles2BackboneCPUAngles_backward, "Angles2Backbone cpu backward for angles");
    m.def("Angles2BackboneCPUParam_backward", &Angles2BackboneCPUParam_backward, "Angles2Backbone cpu backward for parameters");
    m.def("Angles2BackboneGPU_forward", &Angles2BackboneGPU_forward, "Angles2Backbone gpu forward");
    m.def("Angles2BackboneGPUAngles_backward", &Angles2BackboneGPUAngles_backward, "Angles2Backbone gpu backward for angles");
    m.def("Angles2BackboneGPUParam_backward", &Angles2BackboneGPUParam_backward, "Angles2Backbone gpu backward for parameters");
}