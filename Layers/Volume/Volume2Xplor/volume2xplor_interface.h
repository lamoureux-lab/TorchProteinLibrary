#include <torch/extension.h>
void Volume2Xplor(  torch::Tensor volume, const char *filename, float resolution);