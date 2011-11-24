#include <iostream>
#include <ctime>
#include <fstream>
#include "BlackScholes.hpp"
#include "HestonCUDA.hpp"
#include "HestonCUDAPrecision.hpp"
#include "HestonCallFFTCPU.hpp"
#include "HestonCallFFTGPU.hpp"
#include "HestonCallQuadCPU.hpp"
#include "HestonCallQuadGPU.hpp"

void hestonTests() {
	#if defined HestonCUDAPrecisionFloat
		std::cout << "Precision: float" << std::endl;
	#elif defined HestonCUDAPrecisionDouble
		std::cout << "Precision: double" << std::endl;
	#endif

	#ifdef BENCHMARK_RUNS
		std::cout << "Number of runs for benchmark: " << BENCHMARK_RUNS << std::endl;
	#endif

    double fft_price_cpu_1 = HestonCallFFTCPUBenchmark(2,0.04,0.1,0.5,0.04,0.01,0.3,1.0,0.8,4096);
    double fft_price_cpu_2 = HestonCallFFTCPUBenchmark(2,0.04,0.1,0.5,0.04,0.01,1.0,1.0,1.0,4096);
    HestonCUDAPrecision fft_price_gpu_1 = HestonCallFFTGPUBenchmark(2,0.04,0.1,0.5,0.04,0.01,0.3,1.0,0.8,4096);
    HestonCUDAPrecision fft_price_gpu_2 = HestonCallFFTGPUBenchmark(2,0.04,0.1,0.5,0.04,0.01,1.0,1.0,1.0,4096);
    double fft_vol_cpu_1   = BlackScholesImplied(1.0, 0.8, 0.01, 0.3, fft_price_cpu_1);
    double fft_vol_cpu_2   = BlackScholesImplied(1.0, 1.0, 0.01, 1.0, fft_price_cpu_2);
    double fft_price_matlab_1 = 0.203756595588233;
    double fft_price_matlab_2 = 0.085020234245137;
    double fft_vol_matlab_1   = 0.222875129654834;
    double fft_vol_matlab_2   = 0.201741713389921;



    std::cout << "-- fft price --" << std::endl;
    std::cout << "\tCPU\t\t\tMATLAB\t\t\tGPU" << std::endl;
    std::cout << "1\t" << fft_price_cpu_1 << "\t" << fft_price_matlab_1 << "\t" << fft_price_gpu_1 << std::endl;
    std::cout << "2\t" << fft_price_cpu_2 << "\t" << fft_price_matlab_2 << "\t" << fft_price_gpu_2 << std::endl;
    std::cout << "-- fft vol --" << std::endl;
    std::cout << "1\t" << fft_vol_cpu_1   << "\t" << fft_vol_matlab_1 << std::endl;
    std::cout << "2\t" << fft_vol_cpu_2   << "\t" << fft_vol_matlab_2 << std::endl;

    // begin quad tests
    double quad_price_cpu_1 = HestonCallQuadCPUBenchmark(2,.04,.1,0.5,.04,.01,.3,1,.8,4096);
    double quad_price_cpu_2 = HestonCallQuadCPUBenchmark(2,.04,.1,0.5,.04,.01,.3,1,.84,4096);
    double quad_price_matlab_1 = 0.202871648173905;
    double quad_price_matlab_2 = 0.164299483563215;
    double quad_vol_cpu_1 = BlackScholesImplied(1,.80,.01,.3,quad_price_cpu_1);
    double quad_vol_cpu_2 = BlackScholesImplied(1,.84,.01,.3,quad_price_cpu_2);
    double quad_vol_matlab_1 = 0.189204924511565;
    double quad_vol_matlab_2 = 0.191296007291189;
    
    HestonCUDAPrecision quad_price_gpu_1 = HestonCallQuadGPUBenchmark(2,.04,.1,0.5,.04,.01,.3,1,.8,4096);
    HestonCUDAPrecision quad_price_gpu_2 = HestonCallQuadGPUBenchmark(2,.04,.1,0.5,.04,.01,.3,1,.84,4096);
    
	std::cout << "-- quad price --" << std::endl;
    std::cout << "1\t" << quad_price_cpu_1 << "\t" << quad_price_matlab_1 << "\t" << quad_price_gpu_1 << std::endl;
    std::cout << "2\t" << quad_price_cpu_2 << "\t" << quad_price_matlab_2 << "\t" << quad_price_gpu_2 << std::endl;
    std::cout << "-- quad vol --" << std::endl;
    std::cout << "1\t" << quad_vol_cpu_1 << "\t" << quad_vol_matlab_1 << std::endl;
    std::cout << "2\t" << quad_vol_cpu_2 << "\t" << quad_vol_matlab_2 << std::endl;

}

void genFigureData() {
	std::ofstream outfile;
	outfile.open("heston.tab");
	//freopen ("heston.dat","w",stdout);
	outfile << "N\tCPU_QUAD\tGPU_QUAD\tCPU_FFT\tGPU_FFT" << std::endl;
	outfile.precision(12);
	//outfile.setf(std::ios::fixed);
	//outfile.setf(std::ios::showpoint);
	for (int i=8; i<=1048576; i*=2) {
		outfile << i << "\t";

		std::clock_t start = std::clock();
		HestonCallQuadCPUBenchmark(2,.04,.1,0.5,.04,.01,.3,1,.8,i);
		std::clock_t end = std::clock();
		double time =  (end-start)/(double)CLOCKS_PER_SEC;

		outfile << time << "\t";

		start = std::clock();
		HestonCallQuadGPUBenchmark(2,.04,.1,0.5,.04,.01,.3,1,.8,i);
		end   = std::clock();
		time = double (end-start)/CLOCKS_PER_SEC;
		outfile << time << "\t";
		
		start = std::clock();
		HestonCallFFTCPUBenchmark(2,0.04,0.1,0.5,0.04,0.01,0.3,1.0,0.8,i);
		end   = std::clock();
		time = double (end-start)/CLOCKS_PER_SEC;
		outfile << time << "\t";
		
		start = std::clock();
		HestonCallFFTGPUBenchmark(2,0.04,0.1,0.5,0.04,0.01,0.3,1.0,0.8,i);
		end   = std::clock();
		time = double (end-start)/CLOCKS_PER_SEC;
		outfile << time;
		outfile << std::endl;
	}
	outfile.close();
	//std::fclose (stdout);
}

int main(int args, char* argv[]) {
    std::cout.precision(16);
    hestonTests();
	genFigureData();
}

