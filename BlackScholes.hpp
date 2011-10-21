#ifndef __BLACKSCHOLES_HPP__
#define __BLACKSCHOLES_HPP__

double implied(
  double dS0,
  double dK,
  double dR,
  double dT,
  double dCallPrice);

double call(
  double dS0,
  double dK,
  double dR,
  double dT,
  double dSigma);

#endif

