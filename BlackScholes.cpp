#include "BlackScholes.hpp"
#include <math.h>
#include <gsl/gsl_cdf.h>


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
