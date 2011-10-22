#ifndef __BLACKSCHOLES_HPP__
#define __BLACKSCHOLES_HPP__

#define NEWTONS_METHOD_MAX_ERROR  1E-9
#define NEWTONS_METHOD_MAX_ROUNDS 1024

double BlackScholesVega(
  double dS0,
  double dK,
  double dR,
  double dT,
  double dSigma);

double BlackScholesImplied(
  double dS0,
  double dK,
  double dR,
  double dT,
  double dCallPrice);

double BlackScholesCall(
  double dS0,
  double dK,
  double dR,
  double dT,
  double dSigma);

#endif

