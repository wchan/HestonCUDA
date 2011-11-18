#include "HestonCallQuadGPU.hpp"
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

struct HestonCallQuadGPU_functor {
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

	HestonCallQuadGPU_functor(
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
};
HestonCUDAPrecision HestonCallQuadGPU(
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

	return 0;
}

