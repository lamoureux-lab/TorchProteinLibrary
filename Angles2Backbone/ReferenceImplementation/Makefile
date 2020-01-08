NVCC_FLAGS=--std=c++11

main: main.o cBackboneProteinCUDAKernels.o
	nvcc main.o cBackboneProteinCUDAKernels.o -lz -o main

cBackboneProteinCUDAKernels.o: cBackboneProteinCUDAKernels.cu cBackboneProteinCUDAKernels.h cMathCUDAKernels.cu
	nvcc -c --std=c++11 cBackboneProteinCUDAKernels.cu -o cBackboneProteinCUDAKernels.o

main.o: main.cu cBackboneProteinCUDAKernels.h
	nvcc -c --std=c++11 main.cu -o main.o

clean:
	rm -f *.o main
