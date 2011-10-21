#ifndef __HESTONCALLFFTCPU_HPP__
#define __HESTONCALLFFTCPU_HPP__

double HestonCallFFTCPU(
  double dKappa,   // rate of reversion
  double dTheta,   // int run variance
  double dSigma,   // vol of vol
  double dV0,      // initial variance
  double dRho,     // correlation
  double dR,       // instantaneous short rate
  double dT,       // time till maturity
  double dS0,      // initial asset price
  double dStrike,
  long   lN);

#endif

