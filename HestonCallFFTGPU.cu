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
  cuDoubleComplex operator() (int index) {
    cuDoubleComplex zI      = make_cuDoubleComplex(0.0, 1.0);
    HestonCUDAPrecision dU               = index * dEta;

    cuDoubleComplex zV      = make_cuDoubleComplex(dU, -(dAlpha + 1.0));
    cuDoubleComplex zZeta   = mul(-0.5, cuCadd(cuCmul(zV, zV), cuCmul(zI, zV)));
    cuDoubleComplex zGamma  = sub(dKappa, mul(dRho * dSigma, cuCmul(zV, zI)));
    cuDoubleComplex zPHI    = sqrt(cuCsub(cuCmul(zGamma, zGamma), mul(2.0 * dSigma * dSigma, zZeta)));
    
    cuDoubleComplex zA      = mul(dX0 + dR * dT, cuCmul(zI, zV));
    cuDoubleComplex zB      = mul(dV0, cuCdiv(mul(2.0, cuCmul(zZeta, sub(1.0, exp(mul(-dT, zPHI))))), cuCsub(mul(2.0, zPHI), cuCmul(cuCsub(zPHI, zGamma), sub(1.0, exp(mul(-dT, zPHI)))))));
    cuDoubleComplex zC      = mul(-dKappa * dTheta / (dSigma * dSigma), cuCadd(mul(2.0, log(cuCdiv(cuCsub(mul(2.0, zPHI), cuCmul(cuCsub(zPHI, zGamma), sub(1.0, exp(mul(-dT, zPHI))))), (mul(2.0, zPHI))))), mul(dT, cuCsub(zPHI, zGamma))));


    cuDoubleComplex zCharFunc = exp(cuCadd(cuCadd(zA, zB), zC));
    cuDoubleComplex zModifiedCharFunc = cuCdiv(mul(exp(-dR * dT), zCharFunc), add(dAlpha * dAlpha + dAlpha - dU * dU, make_cuDoubleComplex(0.0, dU * (2.0 * dAlpha + 1.0))));

    cuDoubleComplex zSimpsonW = mul(1.0 / 3.0, add(3.0, simpsonWIndex(index)));
    if (index == 0) zSimpsonW.x -= 1.0 / 3.0;

    return mul(dEta, cuCmul(cuCmul(exp(make_cuDoubleComplex(0.0, dB * dU)), zModifiedCharFunc), zSimpsonW));
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

  thrust::copy(dev_zFFTFunc.begin(), dev_zFFTFunc.end(), (cuDoubleComplex*)zFFTFunc);

  cufftHandle p;
#if defined HestonCUDAPrecisionSingle
  cufftSingleComplex* cufftFFTFunc = NULL;
  cufftSingleComplex* cufftPayoff  = NULL;
#elif defined HestonCUDAPrecisionDouble
  cufftDoubleComplex* cufftFFTFunc = NULL;
  cufftDoubleComplex* cufftPayoff  = NULL;
#endif

  cudaMalloc((void**)&cufftFFTFunc, sizeof(HestonCUDAPrecisionComplex) * lN);
  cudaMalloc((void**)&cufftPayoff, sizeof(HestonCUDAPrecisionComplex) * lN);

  cudaMemcpy(cufftFFTFunc, zFFTFunc, sizeof(cufftDoubleComplex) * lN, cudaMemcpyHostToDevice);

#if defined HestonCUDAPrecisionSingle
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

