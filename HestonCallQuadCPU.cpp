#include "HestonCallQuadCPU.hpp"
#define _USE_MATH_DEFINES
#include <cmath>

// erik
// bit of matlab syntax in here still--won't compile (yet)

double HestonCallQuadCPU(
  double kappa,
  double theta,
  double sigma,
	double rho,
  double v0,
  double r,
  double T, 
	double s0,
  double K) {
	
	int N = 0; //unused in the MATLAB code... (and here)
	
	return s0*HestonP(kappa,theta,sigma,rho,v0,r,T,s0,K,1,N) - 
		K*exp(-r*T)*HestonP(kappa,theta,sigma,rho,v0,r,T,s0,K,2,N);

}

double HestonP(double kappa, double theta, double sigma, double rho,
	double v0, double r, double T, double s0, double K, 
	int type, int N) {

	// put HestonPIntegrand in class w/ interface for equation?
	return 0.5 + 1/pi*
		quadl(@HestonPIntegrand,0,100,1e-12,[],kappa, 
		theta,sigma,rho,v0,r,T,s0,K,type);

	
}

double HestonPIntegrand(double phi, double kappa, double theta,
	double sigma, double rho, double v0, double r,
	double T, double s0, double K, int type) {
	
	return real(exp(-i*phi*log(K)).
		*Hestf(phi,kappa,theta,sigma, 
		rho,v0,r,T,s0,type)./(i*phi));

	
}

double HestF(double phi, double kappa, double theta, double sigma,
	double rho, double v0, double r, double T, double s0,
	int type) {
	
	double u, b;
	
	if (type == 1) {
		u = 0.5;
		b = kappa - (rho*sigma);
	} 
	else {
		u = -0.5;
		b = kappa;
	}
	
	double a,x,d,g,C,D,f;
	
	a = kappa*theta;
	x = log(s0);
	d = sqrt((rho*sigma*phi.*i-b).^2-sigma^2*(2*u*phi.*i-phi.^2));
	g = (b-rho*sigma*phi*i + d)./(b-rho*sigma*phi*i - d);
	C = r*phi.*i*T + a/sigma^2.*((b- rho*sigma*phi*i + d)*T - 2*log((1-g.*exp(d*T))./(1-g)));
	D = (b-rho*sigma*phi*i + d)./sigma^2.*((1-exp(d*T))./ (1-g.*exp(d*T)));
	
	f = exp(C + D*v0 + i*phi*x);
	return f;
}

// Lobatto Quadrature
// TODO: returns [Q, fcnt]
double quadl() {
	return 0;
}
