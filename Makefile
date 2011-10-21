CXX      = g++
CXXFLAGS =
LIBS     = -lfftw3 -lm -lgsl -lblas
OBJS     = HestonCallFFTCPU.o BlackScholes.o

all: benchmark

benchmark: ${OBJS} benchmark.cpp
	${CXX} ${CXXFLAGS} -o benchmark benchmark.cpp ${OBJS} ${LIBS}

%.o : %.cpp %.hpp
	${CXX} ${CXXFLAGS} -c $< -o $@

clean:
	rm -rf ${OBJS} benchmark

