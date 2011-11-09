#ifndef __BLACKSCHOLES_HPP__
#define __BLACKSCHOLES_HPP__

#define NEWTONS_METHOD_MAX_ERROR  1E-12
#define NEWTONS_METHOD_MAX_ROUNDS 4096

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

