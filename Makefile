NVCC_FLAGS=--std=c++11

%.o : %.cu
	nvcc -c $(NVCC_FLAGS) $< -o $@


main: main.o cBackboneProteinCUDAKernels.o
	nvcc main.o cBackboneProteinCUDAKernels.o libcnpy.a -lz -o main

clean:
	rm -f *.o main
