#ifndef __HESTONCALLQUADCPU_HPP__
#define __HESTONCALLQUADCPU_HPP__
#include "HestonCUDAPrecision.hpp"
#include <iostream>

double HestonCallQuadCPU(
        double dKappa,
        double dTheta,
        double dSigma,
        double dRho,
        double dV0,
        double dR,
        double dT, 
        double dS0,
        double dK,
        int iN);

#endif

__inline__ double HestonCallQuadCPUBenchmark(
  double dKappa,   // rate of reversion
  double dTheta,   // int run variance
  double dSigma,   // vol of vol
  double dV0,      // initial variance
  double dRho,     // correlation
  double dR,       // instantaneous short rate
  double dT,       // time till maturity
  double dS0,      // initial asset price
  double dStrike,
  long   lN) {
  double result = HestonCallQuadCPU(dKappa, dTheta, dSigma, dV0, dRho, dR, dT, dS0, dStrike, lN);

  clock_t start = clock();
  for (int i = BENCHMARK_RUNS - 1; i >= 0; i--) HestonCallQuadCPU(dKappa, dTheta, dSigma, dV0, dRho, dR, dT, dS0, dStrike, lN);
  clock_t end   = clock();

  std::cout << "CPU Runtime quad(" << BENCHMARK_RUNS << "): " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

  return result;
}
