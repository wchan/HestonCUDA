CXX      = g++
CXXFLAGS = -g
LIBS     = -lfftw3 -lm -lgsl -lblas
OBJS     = HestonCallFFTCPU.o BlackScholes.o HestonCallQuadCPU.o

all: benchmark

benchmark: ${OBJS} benchmark.cpp
	${CXX} ${CXXFLAGS} -o benchmark benchmark.cpp ${OBJS} ${LIBS}

%.o : %.cpp %.hpp
	${CXX} ${CXXFLAGS} -c $< -o $@

clean:
	rm -rf ${OBJS} benchmark

memcheck: benchmark
	valgrind --tool=memcheck --leak-check=full --show-reachable=yes \
	   	./benchmark
