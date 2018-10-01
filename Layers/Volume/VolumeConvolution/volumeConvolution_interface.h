#include <torch/torch.h>
void VolumeConvolution_forward( at::Tensor volume1, 
                                at::Tensor volume2, 
                                at::Tensor output);
void VolumeConvolution_backward(    at::Tensor gradOutput,
                                    at::Tensor gradVolume1,
                                    at::Tensor gradVolume2,
                                    at::Tensor volume1, 
                                    at::Tensor volume2);