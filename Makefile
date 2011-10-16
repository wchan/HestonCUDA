CXX      = g++
CXXFLAGS =
LIBS     = -lfftw3 -lm
OBJS     = HestonCallFFT.o

all: ${OBJS} benchmark

benchmark:
	${CXX} ${CXXFLAGS} -o benchmark benchmark.cpp ${OBJS} ${LIBS}

%.o : %.cpp %.hpp
	${CXX} ${CXXFLAGS} -c $< -o $@

clean:
	rm -rf ${OBJS}

