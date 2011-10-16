#include "HestonCallFFT.hpp"
#include <complex>
#define _USE_MATH_DEFINES
#include <math.h>
#include <fftw3.h>


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
  std::complex<double> zI(0, 1);

  double dX0 = log(dS0);
  double dAlpha = 1.5;
  double dC = 600;
  double dEta = 0.25;
  double dB = M_PI / dEta;

  double vU[lN];
  for (int i = 0; i < lN; i++) vU[i] = i * dEta;

  std::complex<double> zFFTFunc[lN];
  std::complex<double> zPayoff[lN];

  double dLambda = 2 * dB / lN;

  double dPosition = (log(dStrike) + dB) / dLambda + 1;

  for (int i = 0; i < lN; i++) {
    std::complex<double> zV     = vU[i] - (dAlpha + 1.0) * zI;
    std::complex<double> zZeta  = -0.5 * (zV * zV + zI * zV);
    std::complex<double> zGamma = dKappa - dRho * dSigma * zV * zI;
    std::complex<double> zPHI   = sqrt(zGamma * zGamma - 2.0 * dSigma * dSigma * zZeta);
    std::complex<double> zA     = zI * zV * (dX0 + dR * dT);
    std::complex<double> zB     = dV0 * ((2.0 * zZeta * (1.0 - exp(-zPHI * dT))) / (2.0 * zPHI - (zPHI - zGamma) * (1.0 - exp(-zPHI * dT))));
    std::complex<double> zC     = dKappa * dTheta / (dSigma * dSigma) * ( 2.0 * log((2.0 * zPHI - (zPHI - zGamma) * (1.0 - exp(-zPHI * dT))) / ( 2.0 * zPHI)) + (zPHI - zGamma) * dT);

    std::complex<double> zCharFunc = exp(zA + zB + zC);
    std::complex<double> zModifiedCharFunc = zCharFunc * exp(-dR * dT) / (dAlpha * dAlpha + dAlpha - vU[i] * vU[i] + zI * (2.0 + dAlpha + 1.0) * vU[i]);

    std::complex<double> zSimpsonW = 1.0 / 3.0 * (3.0 + pow(zI, i));
    
    if (i == 1) zSimpsonW = zSimpsonW - 1.0 / 3.0;

    zFFTFunc[i] = exp(zI * zB * vU[i]) * zModifiedCharFunc * dEta * zSimpsonW;
  }

  fftw_complex* fftwFFTFunc = reinterpret_cast<fftw_complex*>(zFFTFunc);
  fftw_complex* fftwPayoff  = reinterpret_cast<fftw_complex*>(zPayoff);

  fftw_plan p = fftw_plan_dft_1d(lN, fftwFFTFunc, fftwPayoff, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  std::complex<double> zCallValueM[lN];

  /* wchan: replace this later w/ the appropriate BLAS vector-scalar function */
  for (int i = 0; i < lN; i++) zCallValueM[i] = zPayoff[i] / M_PI;


}

