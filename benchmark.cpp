#include <iostream>
using namespace std;

int main(int args, char* argv[]) {
	hestonTests();
}

void hestonTests() {
	double tol = 1e-6;

	double fft_cuda1 = HestonCallFFT(2,.04,.1,.5,.04,.01,.3,1,.8,4096);
	double fft_matlab1 = 0.203756595588233;
	
	cout << "-- fft_test1 --" << endl;
	cout << "cuda: " << fft_cuda1 << endl;
	cout << "matlab: " << fft_matlab1 << endl;
}

