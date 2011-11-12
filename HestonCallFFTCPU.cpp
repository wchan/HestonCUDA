#include "HestonCUDAPrecision.hpp"
#include "HestonCallFFTCPU.hpp"
#include <complex>
#define _USE_MATH_DEFINES
#include <cmath>
#include <fftw3.h>
#include <gsl/gsl_spline.h>
#include <iostream>

HestonCUDAPrecision HestonCallFFTCPU(
  HestonCUDAPrecision dKappa,   // rate of reversion
  HestonCUDAPrecision dTheta,   // int run variance
  HestonCUDAPrecision dSigma,   // vol of vol
  HestonCUDAPrecision dRho,     // correlation
  HestonCUDAPrecision dV0,      // initial variance
  HestonCUDAPrecision dR,       // instantaneous short rate
  HestonCUDAPrecision dT,       // time till maturity
  HestonCUDAPrecision dS0,      // initial asset price
  HestonCUDAPrecision dStrike,
  long   lN) {
  std::complex<HestonCUDAPrecision> zI(0.0, 1.0);

  HestonCUDAPrecision dX0 = log(dS0);
  HestonCUDAPrecision dAlpha = 1.5;
  HestonCUDAPrecision dC = 600;
  HestonCUDAPrecision dEta = 0.25;
  HestonCUDAPrecision dB = M_PI / dEta;

  HestonCUDAPrecision vU[lN];
  for (int i = 0; i < lN; i++) vU[i] = i * dEta;

  std::complex<HestonCUDAPrecision> zFFTFunc[lN];
  std::complex<HestonCUDAPrecision> zPayoff[lN];
  HestonCUDAPrecision               dPayoff[lN];

  HestonCUDAPrecision dLambda = 2 * dB / lN;
  HestonCUDAPrecision dPosition = (log(dStrike) + dB) / dLambda + 1;

  for (int i = 0; i < lN; i++) {
    std::complex<HestonCUDAPrecision> zV     = vU[i] - (dAlpha + 1.0) * zI;
    std::complex<HestonCUDAPrecision> zZeta  = -0.5 * (zV * zV + zI * zV);
    std::complex<HestonCUDAPrecision> zGamma = dKappa - dRho * dSigma * zV * zI;
    std::complex<HestonCUDAPrecision> zPHI   = sqrt(zGamma * zGamma - 2.0 * dSigma * dSigma * zZeta);

    std::complex<HestonCUDAPrecision> zA     = zI * zV * (dX0 + dR * dT);
    std::complex<HestonCUDAPrecision> zB     = dV0 * ((2.0 * zZeta * (1.0 - exp(-zPHI * dT))) / (2.0 * zPHI - (zPHI - zGamma) * (1.0 - exp(-zPHI * dT))));
    std::complex<HestonCUDAPrecision> zC     = -dKappa * dTheta / (dSigma * dSigma) * ( 2.0 * log((2.0 * zPHI - (zPHI - zGamma) * (1.0 - exp(-zPHI * dT))) / ( 2.0 * zPHI)) + (zPHI - zGamma) * dT);

    std::complex<HestonCUDAPrecision> zCharFunc = exp(zA + zB + zC);
    std::complex<HestonCUDAPrecision> zModifiedCharFunc = zCharFunc * exp(-dR * dT) / (dAlpha * dAlpha + dAlpha - vU[i] * vU[i] + zI * (2.0 * dAlpha + 1.0) * vU[i]);

    std::complex<HestonCUDAPrecision> zSimpsonW = 1.0 / 3.0 * (3.0 + pow(-zI, i + 1));
    
    if (i == 0) zSimpsonW = zSimpsonW - 1.0 / 3.0;

    zFFTFunc[i] = exp(zI * dB * vU[i]) * zModifiedCharFunc * dEta * zSimpsonW;
  }

  fftw_complex* fftwFFTFunc = reinterpret_cast<fftw_complex*>(zFFTFunc);
  fftw_complex* fftwPayoff  = reinterpret_cast<fftw_complex*>(zPayoff);

  fftw_plan p = fftw_plan_dft_1d(lN, fftwFFTFunc, fftwPayoff, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  for (int i = 0; i < lN; i++) dPayoff[i] = zPayoff[i].real();

  HestonCUDAPrecision dCallValueM[lN];

  /* wchan: replace this later w/ the appropriate BLAS vector-scalar function */
  for (int i = 0; i < lN; i++) dCallValueM[i] = dPayoff[i] / M_PI;

  HestonCUDAPrecision dLin[lN];
  for (int i = 0; i < lN; i++) dLin[i] = 1.0 + i;

  gsl_interp_accel* acc = gsl_interp_accel_alloc();
  gsl_spline* spline = gsl_spline_alloc(gsl_interp_cspline, lN);
  gsl_spline_init(spline, dLin, dCallValueM, lN);

  HestonCUDAPrecision dPrice = exp(-log(dStrike) * dAlpha) * gsl_spline_eval(spline, dPosition, acc);

  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);

  return dPrice;
}

