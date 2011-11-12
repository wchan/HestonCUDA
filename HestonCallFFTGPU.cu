#include "HestonCallFFTGPU.hpp"
#include "HestonCUDA.hpp"
#include <complex>
#define _USE_MATH_DEFINES
#include <cmath>
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

__host__ __device__ static __inline__ HestonCUDAPrecisionComplex simpsonWIndex(int index) {
  index &= 3;  
  switch (index) {
    case 0:
      return make_complex(0.0, -1.0);
    case 1:
      return make_complex(-1.0, 0.0);
    case 2:
      return make_complex(0.0, 1.0);
    case 3:
      return make_complex(1.0, 0.0);
  }
  return make_complex(0.0, 0.0);
}

struct HestonCallFFTGPU_functor {
  HestonCUDAPrecision dKappa;
  HestonCUDAPrecision dTheta;
  HestonCUDAPrecision dSigma;
  HestonCUDAPrecision dRho;
  HestonCUDAPrecision dV0;
  HestonCUDAPrecision dR;
  HestonCUDAPrecision dT;
  HestonCUDAPrecision dS0;
  HestonCUDAPrecision dStrike;

  HestonCUDAPrecision dX0;
  HestonCUDAPrecision dAlpha;
  HestonCUDAPrecision dEta;
  HestonCUDAPrecision dB;

  HestonCallFFTGPU_functor(
    HestonCUDAPrecision dKappa,   // rate of reversion
    HestonCUDAPrecision dTheta,   // int run variance
    HestonCUDAPrecision dSigma,   // vol of vol
    HestonCUDAPrecision dRho,     // correlation
    HestonCUDAPrecision dV0,      // initial variance
    HestonCUDAPrecision dR,       // instantaneous short rate
    HestonCUDAPrecision dT,       // time till maturity
    HestonCUDAPrecision dS0,      // initial asset price
    HestonCUDAPrecision dStrike,

    HestonCUDAPrecision dX0,
    HestonCUDAPrecision dAlpha,
    HestonCUDAPrecision dEta,
    HestonCUDAPrecision dB
  ) : dKappa(dKappa), dTheta(dTheta), dSigma(dSigma), dRho(dRho), dV0(dV0), dR(dR), dT(dT), dS0(dS0), dStrike(dStrike), dX0(dX0), dAlpha(dAlpha), dEta(dEta), dB(dB) {}

  __host__ __device__
  HestonCUDAPrecisionComplex operator() (int index) {
    HestonCUDAPrecisionComplex zI      = make_complex(0.0, 1.0);
    HestonCUDAPrecision dU               = index * dEta;

    HestonCUDAPrecisionComplex zV      = make_complex(dU, -(dAlpha + 1.0));
    HestonCUDAPrecisionComplex zZeta   = mul(-0.5, add(mul(zV, zV), mul(zI, zV)));
    HestonCUDAPrecisionComplex zGamma  = sub(dKappa, mul(dRho * dSigma, mul(zV, zI)));
    HestonCUDAPrecisionComplex zPHI    = sqrt(sub(mul(zGamma, zGamma), mul(2.0 * dSigma * dSigma, zZeta)));
    
    HestonCUDAPrecisionComplex zA      = mul(dX0 + dR * dT, mul(zI, zV));
    HestonCUDAPrecisionComplex zB      = mul(dV0, div(mul(2.0, mul(zZeta, sub(1.0, exp(mul(-dT, zPHI))))), sub(mul(2.0, zPHI), mul(sub(zPHI, zGamma), sub(1.0, exp(mul(-dT, zPHI)))))));
    HestonCUDAPrecisionComplex zC      = mul(-dKappa * dTheta / (dSigma * dSigma), add(mul(2.0, log(div(sub(mul(2.0, zPHI), mul(sub(zPHI, zGamma), sub(1.0, exp(mul(-dT, zPHI))))), (mul(2.0, zPHI))))), mul(dT, sub(zPHI, zGamma))));


    HestonCUDAPrecisionComplex zCharFunc = exp(add(add(zA, zB), zC));
    HestonCUDAPrecisionComplex zModifiedCharFunc = div(mul(exp(-dR * dT), zCharFunc), add(dAlpha * dAlpha + dAlpha - dU * dU, make_complex(0.0, dU * (2.0 * dAlpha + 1.0))));

    HestonCUDAPrecisionComplex zSimpsonW = mul(1.0 / 3.0, add(3.0, simpsonWIndex(index)));
    if (index == 0) zSimpsonW.x -= 1.0 / 3.0;

    return mul(dEta, mul(mul(exp(make_complex(0.0, dB * dU)), zModifiedCharFunc), zSimpsonW));
  }
};

HestonCUDAPrecision HestonCallFFTGPU(
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
  // HestonCUDAPrecision dC = 600;
  HestonCUDAPrecision dEta = 0.25;
  HestonCUDAPrecision dB = M_PI / dEta;

  std::complex<HestonCUDAPrecision> zFFTFunc[lN];
  std::complex<HestonCUDAPrecision> zPayoff[lN];
  HestonCUDAPrecision               dPayoff[lN];

  HestonCUDAPrecision dLambda = 2 * dB / lN;
  HestonCUDAPrecision dPosition = (log(dStrike) + dB) / dLambda + 1;

  thrust::device_vector<int> dev_zFFTFuncI(lN);
  thrust::device_vector<HestonCUDAPrecisionComplex> dev_zFFTFunc(lN);
  
  thrust::sequence(dev_zFFTFuncI.begin(), dev_zFFTFuncI.end());
  thrust::transform(dev_zFFTFuncI.begin(), dev_zFFTFuncI.end(), dev_zFFTFunc.begin(), HestonCallFFTGPU_functor(dKappa, dTheta, dSigma, dRho, dV0, dR, dT, dS0, dStrike, dX0, dAlpha, dEta, dB));

  thrust::copy(dev_zFFTFunc.begin(), dev_zFFTFunc.end(), (HestonCUDAPrecisionComplex*)zFFTFunc);

  cufftHandle p;

  HestonCUDAPrecisionComplex* cufftFFTFunc = NULL;
  HestonCUDAPrecisionComplex* cufftPayoff  = NULL;

  cudaMalloc((void**)&cufftFFTFunc, sizeof(HestonCUDAPrecisionComplex) * lN);
  cudaMalloc((void**)&cufftPayoff, sizeof(HestonCUDAPrecisionComplex) * lN);

  cudaMemcpy(cufftFFTFunc, zFFTFunc, sizeof(cufftDoubleComplex) * lN, cudaMemcpyHostToDevice);

#if defined HestonCUDAPrecisionFloat
  cufftPlan1d(&p, lN, CUFFT_C2C, 1);
  cufftExecC2C(p, cufftFFTFunc, cufftPayoff, CUFFT_FORWARD);
#elif defined HestonCUDAPrecisionDouble
  cufftPlan1d(&p, lN, CUFFT_Z2Z, 1);
  cufftExecZ2Z(p, cufftFFTFunc, cufftPayoff, CUFFT_FORWARD);
#endif

  cudaMemcpy(zPayoff, cufftPayoff, sizeof(HestonCUDAPrecisionComplex) * lN, cudaMemcpyDeviceToHost);

  cufftDestroy(p);
  cudaFree(cufftFFTFunc);
  cudaFree(cufftPayoff);


  for (int i = 0; i < lN; i++) dPayoff[i] = zPayoff[i].real();

  double dCallValueM[lN];

  /* wchan: replace this later w/ the appropriate BLAS vector-scalar function */
  for (int i = 0; i < lN; i++) dCallValueM[i] = static_cast<double>(dPayoff[i]) / M_PI;

  double dLin[lN];
  for (int i = 0; i < lN; i++) dLin[i] = 1.0 + i;

  gsl_interp_accel* acc = gsl_interp_accel_alloc();
  gsl_spline* spline = gsl_spline_alloc(gsl_interp_cspline, lN);
  gsl_spline_init(spline, dLin, dCallValueM, lN);

  HestonCUDAPrecision dPrice = exp(-log(dStrike) * dAlpha) * gsl_spline_eval(spline, dPosition, acc);

  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);

  return dPrice;
}

