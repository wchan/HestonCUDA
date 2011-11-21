CXX       = g++
CXXFLAGS  = -O3
LIBS      = -lfftw3 -lm -lblas -lgsl -lcufft
OBJS      = HestonCallFFTCPU.o HestonCallFFTGPU.o BlackScholes.o \
			HestonCallQuadCPU.o HestonCallQuadGPU.o
NVCC      = nvcc
NVCCFLAGS = -O3 --gpu-architecture=compute_20 --use_fast_math \
			--compiler-options "${CXXFLAGS}"

all: benchmark

benchmark: ${OBJS} benchmark.cpp HestonCUDA.hpp
	${NVCC} ${CXXFLAGS} -o benchmark benchmark.cpp ${OBJS} ${LIBS}

%.o: %.cpp %.hpp
	${CXX} ${CXXFLAGS} -c $< -o $@

%.o: %.cu %.hpp
	${NVCC} ${NVCCFLAGS} -c $< -o $@

clean:
	rm -rf ${OBJS} benchmark

memcheck: benchmark
	valgrind --tool=memcheck --leak-check=full --show-reachable=yes \
	   	./benchmark
