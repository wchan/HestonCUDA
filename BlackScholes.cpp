#include "BlackScholes.hpp"
#define _USE_MATH_DEFINES
#include <cmath>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <iostream>


double ugaussian_pdf(double x) {
  return exp(- x * x / 2) / sqrt(2 * M_PI);
}

double BlackScholesCall(
  double dS0,
  double dK,
  double dR,
  double dT,
  double dSigma) {
  double d1 = (log(dS0 / dK) + (dR + dSigma * dSigma / 2.0) * (dT)) / (dSigma * sqrt(dT));
  double d2 = d1 - dSigma * sqrt(dT);

  return gsl_cdf_ugaussian_P(d1) * dS0 - gsl_cdf_ugaussian_P(d2) * dK * exp(- dR * dT);
}

double BlackScholesVega(
  double dS0,
  double dK,
  double dR,
  double dT,
  double dSigma) {
  double d1 = (log(dS0 / dK) + (dR + dSigma * dSigma / 2.0) * (dT)) / (dSigma * sqrt(dT));

  return dS0 * gsl_ran_ugaussian_pdf(d1) * sqrt(dT);
  //return dS0 * ugaussian_pdf(d1) * sqrt(dT);
}

double BlackScholesImplied(
  double dS0,
  double dK,
  double dR,
  double dT,
  double dCallPrice) {
  /* use newton's method... */
  double dVol = 0.50;
  double dVolOld;
  double dError;
  long lRounds = 0;

  do {
    dVolOld = dVol;
    
    dVol -= (BlackScholesCall(dS0, dK, dR, dT, dVol) - dCallPrice) / BlackScholesVega(dS0, dK, dR, dT, dVol);

    dError = std::abs(dVol - dVolOld);
  } while (dError > NEWTONS_METHOD_MAX_ERROR && lRounds++ < NEWTONS_METHOD_MAX_ROUNDS);

  return dVol;
}

