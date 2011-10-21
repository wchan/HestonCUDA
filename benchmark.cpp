#include <iostream>
#include "HestonCallFFT.hpp"


int main(int args, char* argv[]) {
  double mat = 1;
  double strike = 1;

  std::cout << "Call Price: " << HestonCallFFT(2,.04,.1,0.5,.04,.01,mat,1,strike,4096) << std::endl;
}

