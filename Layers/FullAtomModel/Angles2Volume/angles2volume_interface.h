void Angles2Volume_forward( const char* sequence,THDoubleTensor *input_angles, THCudaTensor *output_volume,int add_terminal,THDoubleTensor *R_ext);
void Angles2Volume_backward( const char* sequence,THDoubleTensor *input_angles, THCudaTensor *grad_volume,THDoubleTensor *grad_angles,int add_terminal,THDoubleTensor *R_ext,THDoubleTensor *T_ext);
