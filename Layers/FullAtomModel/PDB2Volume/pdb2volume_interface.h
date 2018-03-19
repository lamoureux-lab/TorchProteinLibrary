void PDB2Volume( THByteTensor *filenames, THFloatTensor *volume);
void PDB2VolumeCUDA( THByteTensor *filenames, THCudaTensor *volume, int rotate, int translate);