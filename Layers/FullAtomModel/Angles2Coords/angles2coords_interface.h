void Angles2Coords_forward(  const char* sequence,THDoubleTensor *input_angles, THDoubleTensor *output_coords);
void Angles2Coords_backward( THDoubleTensor *grad_atoms,THDoubleTensor *grad_angles,const char* sequence,THDoubleTensor *input_angles);
void Angles2Coords_save(  const char* sequence,THDoubleTensor *input_angles, const char* output_filename);