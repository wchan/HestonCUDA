CXX       = g++
CXXFLAGS  = -O3
LIBS      = -lfftw3 -lm -lgsl -lblas -lcufft
OBJS      = HestonCallFFTCPU.o HestonCallFFTGPU.o BlackScholes.o HestonCallQuadCPU.o
NVCC      = nvcc
NVCCFLAGS = --gpu-architecture=compute_20

all: benchmark

benchmark: ${OBJS} benchmark.cpp
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
