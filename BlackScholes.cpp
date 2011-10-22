#include "BlackScholes.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>


double ugaussian_pdf(double x) {
  return exp(- x * x / 2) / sqrt(2 * M_PI);
}

double implied(
  double dS0,
  double dK,
  double dR,
  double dT,
  double dCallPrice) {
  return 0;
}

double call(
  double dS0,
  double dK,
  double dR,
  double dT,
  double dSigma) {
  double dF = dS0 * exp(dR * dT);
  double d1 = log(dF / dK) / (dSigma * sqrt(dT)) + dSigma * sqrt(dT) / 2;
  double d2 = log(dF / dK) / (dSigma * sqrt(dT)) - dSigma * sqrt(dT) / 2;

  return exp(-dR * dT) * (dF * gsl_cdf_ugaussian_P(d1) - dK * gsl_cdf_ugaussian_P(d2));
}

double vega(
  double dS0,
  double dK,
  double dR,
  double dT,
  double dSigma) {
  double dF = dS0 * exp(dR * dT);
  double d1 = log(dF / dK) / (dSigma * sqrt(dT)) + dSigma * sqrt(dT) / 2;

  // return dS0 * gsl_ran_ugaussian_pdf(d1) * sqrt(dT);
  return dS0 * ugaussian_pdf(d1) * sqrt(dT);
}

