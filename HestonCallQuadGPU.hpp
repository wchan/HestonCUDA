#ifndef __HESTONCALLQUADGPU_HPP__
#define __HESTONCALLQUADGPU_HPP__

#include "HestonCUDA.hpp"
#include <iostream>


HestonCUDAPrecision HestonCallQuadGPU(
  HestonCUDAPrecision dKappa,   // rate of reversion
  HestonCUDAPrecision dTheta,   // int run variance
  HestonCUDAPrecision dSigma,   // vol of vol
  HestonCUDAPrecision dV0,      // initial variance
  HestonCUDAPrecision dRho,     // correlation
  HestonCUDAPrecision dR,       // instantaneous short rate
  HestonCUDAPrecision dT,       // time till maturity
  HestonCUDAPrecision dS0,      // initial asset price
  HestonCUDAPrecision dStrike,
  long   lN);

__inline__ HestonCUDAPrecision HestonCallQuadGPUBenchmark(
  HestonCUDAPrecision dKappa,   // rate of reversion
  HestonCUDAPrecision dTheta,   // int run variance
  HestonCUDAPrecision dSigma,   // vol of vol
  HestonCUDAPrecision dV0,      // initial variance
  HestonCUDAPrecision dRho,     // correlation
  HestonCUDAPrecision dR,       // instantaneous short rate
  HestonCUDAPrecision dT,       // time till maturity
  HestonCUDAPrecision dS0,      // initial asset price
  HestonCUDAPrecision dStrike,
  long   lN) {
  HestonCUDAPrecision result = HestonCallQuadGPU(dKappa, dTheta, dSigma, dV0, dRho, dR, dT, dS0, dStrike, lN);

  clock_t start = clock();
  for (int i = BENCHMARK_RUNS - 1; i >= 0; i--) HestonCallQuadGPU(dKappa, dTheta, dSigma, dV0, dRho, dR, dT, dS0, dStrike, lN);
  clock_t end   = clock();

  std::cout << "GPU Runtime quad(" << BENCHMARK_RUNS << "): " << (HestonCUDAPrecision)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

  return result;
}

#endif

