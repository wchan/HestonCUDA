#ifndef __HESTONCALLFFT_HPP__
#define __HESTONCALLFFT_HPP__

#include "HestonCallFFTCPU.hpp"

double HestonCallFFT(
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
  return HestonCallFFTCPU(dKappa, dTheta, dSigma, dV0, dRho, dR, dT, dS0, dStrike, lN);
}

#endif

