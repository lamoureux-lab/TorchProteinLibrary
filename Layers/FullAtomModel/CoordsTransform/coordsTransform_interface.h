void CoordsTranslate_forward(   THDoubleTensor *input_coords, THDoubleTensor *output_coords,THDoubleTensor *T,THIntTensor *num_atoms);
void CoordsTranslate_backward(  THDoubleTensor *grad_output_coords, THDoubleTensor *grad_input_coords);
void CoordsRotate_forward(  THDoubleTensor *input_coords, THDoubleTensor *output_coords,THDoubleTensor *R,THIntTensor *num_atoms);
void CoordsRotate_backward( THDoubleTensor *grad_output_coords, THDoubleTensor *grad_input_coords,THDoubleTensor *R,THIntTensor *num_atoms);
void getBBox( THDoubleTensor *input_coords,THDoubleTensor *a, THDoubleTensor *b,THIntTensor *num_atoms);
void getRandomRotation( THDoubleTensor *R );
void getRandomTranslation( THDoubleTensor *T, THDoubleTensor *a, THDoubleTensor *b, int volume_size );