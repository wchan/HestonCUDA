#ifndef __HESTON_CUDA__
#define __HESTON_CUDA__

#include "HestonCallFFTCPU.hpp"
#include "HestonCallFFTGPU.hpp"

#include <cuComplex.h>


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
  return make_cuDoubleComplex(s - c.x, -c.y);
}

__host__ __device__ static __inline__ cuDoubleComplex add(double s, cuDoubleComplex c) {
  return make_cuDoubleComplex(s + c.x, c.y);
}

__host__ __device__ static __inline__ cuDoubleComplex sqrt(cuDoubleComplex c) {
  double f = sqrt(mag(c));
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

#endif

