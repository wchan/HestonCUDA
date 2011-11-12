#include "HestonCallFFTGPU.hpp"
#include <complex>
#define _USE_MATH_DEFINES
#include <cmath>
#include <fftw3.h>
#include <gsl/gsl_spline.h>
#include <iostream>

// NVIDIA CUDA Headers
#include <cuda.h>
#include <cuComplex.h>

// NVIDIA Thrust Headers (http://developer.nvidia.com/Thrust)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>

// NVIDIA CUFFT
#include <cufft.h>

__host__ __device__ static __inline__ double mag(cuDoubleComplex c) {
  return sqrt(c.x * c.x + c.y * c.y);
}

__host__ __device__ static __inline__ double phase(cuDoubleComplex c) {
  return atan(c.y / c.x);
}

__host__ __device__ static __inline__ cuDoubleComplex mul(double s, cuDoubleComplex c) {
  return make_cuDoubleComplex(s * c.x, s * c.y);
}

__host__ __device__ static __inline__ cuDoubleComplex sub(double s, cuDoubleComplex c) {
  return make_cuDoubleComplex(s - c.x, c.y);
}

__host__ __device__ static __inline__ cuDoubleComplex add(double s, cuDoubleComplex c) {
  return make_cuDoubleComplex(s + c.x, c.y);
}

__host__ __device__ static __inline__ cuDoubleComplex sqrt(cuDoubleComplex c) {
  double f = mag(c);
  double hp = 0.5 * phase(c);
  
  return make_cuDoubleComplex(f * cos(hp), f * sin(hp));
}

__host__ __device__ static __inline__ cuDoubleComplex exp(cuDoubleComplex c) {
  double f = exp(c.x);

  return make_cuDoubleComplex(f * cos(c.y), f * sin(c.y));
}

__host__ __device__ static __inline__ cuDoubleComplex log(cuDoubleComplex c) {
  return make_cuDoubleComplex(log(mag(c)), phase(c));
}

__host__ __device__ static __inline__ cuDoubleComplex simpsonWIndex(int index) {
  index &= 3;  
  switch (index) {
    case 0:
      return make_cuDoubleComplex(0.0, -1.0);
    case 1:
      return make_cuDoubleComplex(-1.0, 0.0);
    case 2:
      return make_cuDoubleComplex(0.0, 1.0);
    case 3:
      return make_cuDoubleComplex(1.0, 0.0);
  }
  return make_cuDoubleComplex(0.0, 0.0);
}

struct HestonCallFFTGPU_functor {
  double dKappa;
  double dTheta;
  double dSigma;
  double dRho;
  double dV0;
  double dR;
  double dT;
  double dS0;
  double dStrike;

  double dX0;
  double dAlpha;
  double dEta;
  double dB;

  HestonCallFFTGPU_functor(
    double dKappa,   // rate of reversion
    double dTheta,   // int run variance
    double dSigma,   // vol of vol
    double dRho,     // correlation
    double dV0,      // initial variance
    double dR,       // instantaneous short rate
    double dT,       // time till maturity
    double dS0,      // initial asset price
    double dStrike,

    double dX0,
    double dAlpha,
    double dEta,
    double dB
  ) : dKappa(dKappa), dTheta(dTheta), dSigma(dSigma), dRho(dRho), dV0(dV0), dR(dR), dT(dT), dS0(dS0), dStrike(dStrike), dX0(dX0), dAlpha(dAlpha), dEta(dEta), dB(dB) {}

  __host__ __device__
  cuDoubleComplex operator() (int index) {
    cuDoubleComplex zI      = make_cuDoubleComplex(0.0, 1.0);

    double dU               = index * dEta;
    cuDoubleComplex zV      = make_cuDoubleComplex(dU, dAlpha + 1.0);
    cuDoubleComplex zZeta   = mul(0.5, cuCadd(cuCmul(zV, zV), cuCmul(zI, zV)));
    cuDoubleComplex zGamma  = sub(dKappa, mul(dRho * dSigma, cuCmul(zV, zI)));
    cuDoubleComplex zPHI    = sqrt(cuCsub(cuCmul(zGamma, zGamma), mul(2.0 * dSigma * dSigma, zZeta)));
    cuDoubleComplex zA      = mul(dX0 + dR * dT, cuCmul(zI, zV));
    cuDoubleComplex zB      = mul(dV0, cuCdiv(mul(2.0, cuCmul(zZeta, sub(1, exp(mul(-dT, zPHI))))), cuCsub(mul(2.0, zPHI), cuCmul(cuCsub(zPHI, zGamma), sub(1.0, exp(mul(-dT, zPHI)))))));
    cuDoubleComplex zC      = mul(-dKappa * dTheta / (dSigma * dSigma), cuCadd(mul(2.0, log(cuCdiv(cuCsub(mul(2.0, zPHI), cuCmul(cuCsub(zPHI, zGamma), sub(1.0, exp(mul(-dT, zPHI))))), (mul(2.0, zPHI))))), mul(dT, cuCsub(zPHI, zGamma))));


    cuDoubleComplex zCharFunc = exp(cuCadd(cuCadd(zA, zB), zC));
    cuDoubleComplex zModifiedCharFunc = cuCdiv(mul(exp(-dR * dT), zCharFunc), add(dAlpha * dAlpha + dAlpha - dU * dU, make_cuDoubleComplex(0.0, dU * (2.0 * dAlpha + 1.0))));

    cuDoubleComplex zSimpsonW = mul(1.0 / 3.0, add(3.0, simpsonWIndex(index)));
    if (index == 0) zSimpsonW.x -= 1.0 / 3.0;

    return mul(dEta, cuCmul(cuCmul(exp(make_cuDoubleComplex(0.0, dB * dU)), zModifiedCharFunc), zSimpsonW));
  }
};

double HestonCallFFTGPU(
  double dKappa,   // rate of reversion
  double dTheta,   // int run variance
  double dSigma,   // vol of vol
  double dRho,     // correlation
  double dV0,      // initial variance
  double dR,       // instantaneous short rate
  double dT,       // time till maturity
  double dS0,      // initial asset price
  double dStrike,
  long   lN) {
  std::complex<double> zI(0.0, 1.0);

  double dX0 = log(dS0);
  double dAlpha = 1.5;
  // double dC = 600;
  double dEta = 0.25;
  double dB = M_PI / dEta;

  std::complex<double> zFFTFunc[lN];
  std::complex<double> zPayoff[lN];
  double               dPayoff[lN];

  double dLambda = 2 * dB / lN;
  double dPosition = (log(dStrike) + dB) / dLambda + 1;
  
  thrust::device_vector<int> dev_zFFTFuncI(lN);
  thrust::device_vector<cuDoubleComplex> dev_zFFTFunc(lN);
  
  thrust::sequence(dev_zFFTFuncI.begin(), dev_zFFTFuncI.end());
  thrust::transform(dev_zFFTFuncI.begin(), dev_zFFTFuncI.end(), dev_zFFTFunc.begin(), HestonCallFFTGPU_functor(dKappa, dTheta, dSigma, dRho, dV0, dR, dT, dS0, dStrike, dX0, dAlpha, dEta, dB));

  thrust::copy(dev_zFFTFunc.begin(), dev_zFFTFunc.end(), (cuDoubleComplex*)zFFTFunc);

  fftw_complex* fftwFFTFunc = reinterpret_cast<fftw_complex*>(zFFTFunc);
  fftw_complex* fftwPayoff  = reinterpret_cast<fftw_complex*>(zPayoff);

  fftw_plan p = fftw_plan_dft_1d(lN, fftwFFTFunc, fftwPayoff, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
  
  /*
  fftw_complex* fftwFFTFunc = reinterpret_cast<fftw_complex*>(zFFTFunc);
  fftw_complex* fftwPayoff  = reinterpret_cast<fftw_complex*>(zPayoff);

  fftw_plan p = fftw_plan_dft_1d(lN, fftwFFTFunc, fftwPayoff, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
  */

  for (int i = 0; i < lN; i++) dPayoff[i] = zPayoff[i].real();

  double dCallValueM[lN];

  /* wchan: replace this later w/ the appropriate BLAS vector-scalar function */
  for (int i = 0; i < lN; i++) dCallValueM[i] = dPayoff[i] / M_PI;

  double dLin[lN];
  for (int i = 0; i < lN; i++) dLin[i] = 1.0 + i;

  gsl_interp_accel* acc = gsl_interp_accel_alloc();
  gsl_spline* spline = gsl_spline_alloc(gsl_interp_cspline, lN);
  gsl_spline_init(spline, dLin, dCallValueM, lN);

  double dPrice = exp(-log(dStrike) * dAlpha) * gsl_spline_eval(spline, dPosition, acc);

  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);

  return dPrice;
}

