CXX      = g++
CXXFLAGS = -g
LIBS     = -lfftw3 -lm -lgsl -lblas
OBJS     = HestonCallFFTCPU.o HestonCallFFTGPU.o BlackScholes.o HestonCallQuadCPU.o
NVCC     = nvcc

all: benchmark

benchmark: ${OBJS} benchmark.cpp
	${CXX} ${CXXFLAGS} -o benchmark benchmark.cpp ${OBJS} ${LIBS}

%.o: %.cpp %.hpp
	${CXX} ${CXXFLAGS} -c $< -o $@

%.o: %.cu %.hpp
	${NVCC} ${CXXFLAGS} -c $< -o $@

clean:
	rm -rf ${OBJS} benchmark

memcheck: benchmark
	valgrind --tool=memcheck --leak-check=full --show-reachable=yes \
	   	./benchmark
