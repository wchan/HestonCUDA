#ifndef __HESTONCALLFFTCPU_HPP__
#define __HESTONCALLFFTCPU_HPP__

#include "HestonCUDAPrecision.hpp"
#include <iostream>


HestonCUDAPrecision HestonCallFFTCPU(
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

__inline__ HestonCUDAPrecision HestonCallFFTCPUBenchmark(
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
  HestonCUDAPrecision result = HestonCallFFTCPU(dKappa, dTheta, dSigma, dV0, dRho, dR, dT, dS0, dStrike, lN);

  clock_t start = clock();
  for (int i = BENCHMARK_RUNS - 1; i >= 0; i--) HestonCallFFTCPU(dKappa, dTheta, dSigma, dV0, dRho, dR, dT, dS0, dStrike, lN);
  clock_t end   = clock();

  std::cout << "CPU Runtime (" << BENCHMARK_RUNS << "): " << (HestonCUDAPrecision)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

  return result;
}

#endif

