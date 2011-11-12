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

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex add(HestonCUDAPrecisionComplex x, HestonCUDAPrecisionComplex y) {
#if defined HestonCUDAPrecisionFloat
  return cuCaddf(x, y);
#elif defined HestonCUDAPrecisionDouble
  return cuCadd(x, y);
#endif
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex sub(HestonCUDAPrecisionComplex x, HestonCUDAPrecisionComplex y) {
#if defined HestonCUDAPrecisionFloat
  return cuCsubf(x, y);
#elif defined HestonCUDAPrecisionDouble
  return cuCsub(x, y);
#endif
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex mul(HestonCUDAPrecisionComplex x, HestonCUDAPrecisionComplex y) {
#if defined HestonCUDAPrecisionFloat
  return cuCmulf(x, y);
#elif defined HestonCUDAPrecisionDouble
  return cuCmul(x, y);
#endif
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex div(HestonCUDAPrecisionComplex x, HestonCUDAPrecisionComplex y) {
#if defined HestonCUDAPrecisionFloat
  return cuCdivf(x, y);
#elif defined HestonCUDAPrecisionDouble
  return cuCdiv(x, y);
#endif
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex make_complex(HestonCUDAPrecision r, HestonCUDAPrecision i) {
#if defined HestonCUDAPrecisionFloat
  return make_cuFloatComplex(r, i);
#elif defined HestonCUDAPrecisionDouble
  return make_cuDoubleComplex(r, i);
#endif
}

__host__ __device__ static __inline__ HestonCUDAPrecision mag(HestonCUDAPrecisionComplex c) {
  return sqrt(c.x * c.x + c.y * c.y);
}

__host__ __device__ static __inline__ HestonCUDAPrecision phase(HestonCUDAPrecisionComplex c) {
  return atan(c.y / c.x);
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex mul(HestonCUDAPrecision s, HestonCUDAPrecisionComplex c) {
  return make_complex(s * c.x, s * c.y);
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex sub(HestonCUDAPrecision s, HestonCUDAPrecisionComplex c) {
  return make_complex(s - c.x, -c.y);
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex add(HestonCUDAPrecision s, HestonCUDAPrecisionComplex c) {
  return make_complex(s + c.x, c.y);
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex sqrt(HestonCUDAPrecisionComplex c) {
  HestonCUDAPrecision f = sqrt(mag(c));
  HestonCUDAPrecision hp = 0.5 * phase(c);
  
  return make_complex(f * cos(hp), f * sin(hp));
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex exp(HestonCUDAPrecisionComplex c) {
  HestonCUDAPrecision f = exp(c.x);

  return make_complex(f * cos(c.y), f * sin(c.y));
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex log(HestonCUDAPrecisionComplex c) {
  return make_complex(log(mag(c)), phase(c));
}

#endif
#endif

