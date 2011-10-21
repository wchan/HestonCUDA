#include <iostream>
#include "HestonCallFFT.hpp"


void hestonTests() {
	double tol = 1e-6;

	double fft_cpu1 = HestonCallFFT(2,0.04,0.1,0.5,0.04,0.01,0.3,1.0,0.8,4096);
  double fft_cpu2 = HestonCallFFT(2,0.04,0.1,0.5,0.04,0.01,1.0,1.0,1.0,4096);
	double fft_matlab1 = 0.203756595588233;
  double fft_matlab2 = 0.085020234245137;
	
	std::cout << "-- fft heston call --" << std::endl;
  std::cout << "\tCPU\t\t\tMATLAB" << std::endl;
  std::cout << "1\t" << fft_cpu1 << "\t" << fft_matlab1 << std::endl;
  std::cout << "2\t" << fft_cpu2 << "\t" << fft_matlab2 << std::endl;
}

int main(int args, char* argv[]) {
	std::cout.precision(16);
  hestonTests();
}

