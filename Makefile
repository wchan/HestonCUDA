CXX      = g++
CXXFLAGS =
LIBS     = -lfftw3 -lm -lgsl -lblas
OBJS     = HestonCallFFT.o

all: ${OBJS} benchmark

benchmark: benchmark.cpp
	${CXX} ${CXXFLAGS} -o benchmark benchmark.cpp ${OBJS} ${LIBS}

%.o : %.cpp %.hpp
	${CXX} ${CXXFLAGS} -c $< -o $@

clean:
	rm -rf ${OBJS}

