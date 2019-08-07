#include <torch/extension.h>
void VolumeConvolution_forward( torch::Tensor volume1, 
                                torch::Tensor volume2, 
                                torch::Tensor output);
void VolumeConvolution_backward(    torch::Tensor gradOutput,
                                    torch::Tensor gradVolume1,
                                    torch::Tensor gradVolume2,
                                    torch::Tensor volume1, 
                                    torch::Tensor volume2);