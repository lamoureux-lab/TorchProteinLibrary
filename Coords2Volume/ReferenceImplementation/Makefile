NVCC_FLAGS=--std=c++11

main: main.o Kernel.o HashKernel.o
	nvcc main.o Kernel.o HashKernel.o -lz -o main

HashKernel.o: HashKernel.cu
	nvcc -c --std=c++11 HashKernel.cu -o HashKernel.o

Kernel.o: Kernel.cu HashKernel.cu
	nvcc -c --std=c++11 Kernel.cu -o Kernel.o

main.o: main.cu Kernel.h HashKernel.h
	nvcc -c --std=c++11 main.cu -o main.o

clean:
	rm -f *.o main