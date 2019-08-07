#include <torch/extension.h>
#include <Angles2Backbone/angles2backbone_interface.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("Angles2BackboneCPU_forward", &Angles2BackboneCPU_forward, "Angles2Backbone cpu forward");
    m.def("Angles2BackboneCPU_backward", &Angles2BackboneCPU_backward, "Angles2Backbone cpu backward");
    m.def("Angles2BackboneGPU_forward", &Angles2BackboneGPU_forward, "Angles2Backbone gpu forward");
    m.def("Angles2BackboneGPU_backward", &Angles2BackboneGPU_backward, "Angles2Backbone gpu backward");
}