#ifndef __HESTONCUDAPRECISION_HPP__
#define __HESTONCUDAPRECISION_HPP__

#define HestonCUDAPrecisionDouble
#if defined HestonCUDAPrecisionFloat
	#define HestonCUDAPrecision float
	#define HestonCUDAPrecisionComplex cuFloatComplex
#elif defined HestonCUDAPrecisionDouble
	#define HestonCUDAPrecision double
	#define HestonCUDAPrecisionComplex cuDoubleComplex
#else
	#error Please Specify HestonCUDAPrecision
#endif

#define BENCHMARK_RUNS 1

#endif

