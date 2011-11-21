#ifndef __HESTONCUDA_HPP__
#define __HESTONCUDA_HPP__

#include "HestonCUDAPrecision.hpp"
#include <cuda.h>
#include <cuComplex.h>

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

// with parameter types flipped
__host__ __device__ static __inline__ HestonCUDAPrecisionComplex mul(HestonCUDAPrecisionComplex s, HestonCUDAPrecision c) {
  return mul(c,s);
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex sub(HestonCUDAPrecisionComplex s, HestonCUDAPrecision c) {
  return make_complex(s.x - c, s.y);
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex add(HestonCUDAPrecisionComplex s, HestonCUDAPrecision c) {
  return add(c,s);
}

// operator overloading... probably a cleaner way to do this w/ templates
__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator+(const HestonCUDAPrecision& lhs, const HestonCUDAPrecisionComplex& rhs) {
	return add(lhs,rhs);
}

__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator+(const HestonCUDAPrecisionComplex& lhs, const HestonCUDAPrecision& rhs) {
	return add(lhs,rhs);
}

__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator+(const HestonCUDAPrecisionComplex& lhs, const HestonCUDAPrecisionComplex& rhs) {
	return add(lhs,rhs);
}

__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator-(const HestonCUDAPrecision& lhs, const HestonCUDAPrecisionComplex& rhs) {
	return sub(lhs,rhs);
}

__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator-(const HestonCUDAPrecisionComplex& lhs, const HestonCUDAPrecision& rhs) {
	return sub(lhs,rhs);
}

__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator-(const HestonCUDAPrecisionComplex& lhs, const HestonCUDAPrecisionComplex& rhs) {
	return sub(lhs,rhs);
}

__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator-(const HestonCUDAPrecisionComplex& lhs) {
	return make_complex(-lhs.x,-lhs.y);
}

__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator*(const HestonCUDAPrecision& lhs, const HestonCUDAPrecisionComplex& rhs) {
	return mul(lhs,rhs);
}

__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator*(const HestonCUDAPrecisionComplex& lhs, const HestonCUDAPrecision& rhs) {
	return mul(lhs,rhs);
}

__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator*(const HestonCUDAPrecisionComplex& lhs, const HestonCUDAPrecisionComplex& rhs) {
	return mul(lhs,rhs);
}

__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator/(const HestonCUDAPrecision& lhs, const HestonCUDAPrecisionComplex& rhs) {
	return div(make_complex(lhs,0),rhs);
}

__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator/(const HestonCUDAPrecisionComplex& lhs, const HestonCUDAPrecision& rhs) {
	return div(lhs,make_complex(rhs,0));
}

__host__ __device__ static __inline__ const
HestonCUDAPrecisionComplex operator/(const HestonCUDAPrecisionComplex& lhs, const HestonCUDAPrecisionComplex& rhs) {
	return div(lhs,rhs);
}
// end operator overloading

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex sqrt(HestonCUDAPrecisionComplex c) {
  HestonCUDAPrecision f = sqrt(mag(c));
  HestonCUDAPrecision hp = 0.5 * phase(c);
  
  return make_complex(f * cos(hp),f * sin(hp));
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex exp(HestonCUDAPrecisionComplex c) {
//  HestonCUDAPrecision f = exp(c.x);
//
//  return make_complex(f * cos(c.y), f * sin(c.y));

	HestonCUDAPrecisionComplex res;
	HestonCUDAPrecision t = exp (c.x);
	sincos (c.y, &res.y, &res.x);
	res.x *= t;
	res.y *= t;
	return res;

}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex log(HestonCUDAPrecisionComplex c) {
  return make_complex(log(mag(c)), phase(c));
}

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex pow(HestonCUDAPrecisionComplex c, int exponent) {
  return exp(mul(exponent,log(c)));
}

#endif
#endif

