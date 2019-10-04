#include <torch/extension.h>
#include "coords2eps_interface.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test", &test, "test function");
}