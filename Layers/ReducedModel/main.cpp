#include <torch/torch.h>
#include <Angles2Backbone/angles2backbone_interface.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("Angles2Backbone_forward", &Angles2Backbone_forward, "Angles2Backbone forward");
    m.def("Angles2Backbone_backward", &Angles2Backbone_backward, "Angles2Backbone backward");
}