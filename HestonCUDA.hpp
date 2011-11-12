#ifndef __HESTONCUDA_HPP__
#define __HESTONCUDA_HPP__

#include "HestonCUDAPrecision.hpp"
#include <cuda.h>
#include <cuComplex.h>


/*
__inline__ HestonCUDAPrecision HestonCallFFT(
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
  return HestonCallFFTCPU(dKappa, dTheta, dSigma, dV0, dRho, dR, dT, dS0, dStrike, lN);
}
*/

#ifdef __CUDACC__

__host__ __device__ static __inline__ HestonCUDAPrecision mag(cuDoubleComplex c) {
  return sqrt(c.x * c.x + c.y * c.y);
}

__host__ __device__ static __inline__ HestonCUDAPrecision phase(cuDoubleComplex c) {
  return atan(c.y / c.x);
}

__host__ __device__ static __inline__ cuDoubleComplex mul(HestonCUDAPrecision s, cuDoubleComplex c) {
  return make_cuDoubleComplex(s * c.x, s * c.y);
}

__host__ __device__ static __inline__ cuDoubleComplex sub(HestonCUDAPrecision s, cuDoubleComplex c) {
  return make_cuDoubleComplex(s - c.x, -c.y);
}

__host__ __device__ static __inline__ cuDoubleComplex add(HestonCUDAPrecision s, cuDoubleComplex c) {
  return make_cuDoubleComplex(s + c.x, c.y);
}

__host__ __device__ static __inline__ cuDoubleComplex sqrt(cuDoubleComplex c) {
  HestonCUDAPrecision f = sqrt(mag(c));
  HestonCUDAPrecision hp = 0.5 * phase(c);
  
  return make_cuDoubleComplex(f * cos(hp), f * sin(hp));
}

__host__ __device__ static __inline__ cuDoubleComplex exp(cuDoubleComplex c) {
  HestonCUDAPrecision f = exp(c.x);

  return make_cuDoubleComplex(f * cos(c.y), f * sin(c.y));
}

__host__ __device__ static __inline__ cuDoubleComplex log(cuDoubleComplex c) {
  return make_cuDoubleComplex(log(mag(c)), phase(c));
}

#endif
#endif

