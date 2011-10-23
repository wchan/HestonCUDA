#include "HestonCallQuadCPU.hpp"
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

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

		// previously, the arg @HestonPIntegrand was passed to quadl,
		// which specified the quadrature function.
		// HestonPIntegrand is now coded into quadl as the function
		// since it is the only function used for quadl in our code.
		
		return 0.5 + 1/pi*
				quadl(@HestonPIntegrand,0,100,1e-12,[],kappa, 
								theta,sigma,rho,v0,r,T,s0,K,type);


}

double HestonPIntegrand(double* phi, int phi_size, double kappa, double theta,
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


void legendre(int ind, double* quad_1d_point, 
				double* quad_1d_weight) {
		if (ind == 1) {
				*quad_1d_point = 0;
				*quad_1d_weight = 2;
		}
		else if (ind == 2) {
				*quad_1d_point = {-1.0/sqrt(3.0), 1.0/sqrt(3.0)};
				*quad_1d_weight = {1, 1};
		}
		else if (ind == 3) {
				*quad_1d_point = {-sqrt(3.0/5.0), 0, sqrt(3.0/5.0)};
				*quad_1d_weight = {5.0/9.0, 8.0/9.0, 5.0/9.0};
		}
		else if (ind == 4) {
				double a,b,w1,w2;
				a = sqrt((3.0+2.0*sqrt(6.0/5.0)/7.0));
				b = sqrt((3.0-2.0*sqrt(6.0/5.0)/7.0));
				w1 = (18.0-sqrt(30.0))/36.0;
				w2 = (18.0+sqrt(30.0))/36.0;
				*quad_1d_point = {-a, -b, b, a};
				*quad_1d_weight = {w1, w2, w2, w1};
		}
		else if (ind == 5) {
				double a,b,w1,w2;
				a = 1.0/3.0*sqrt(5.0+2.0*sqrt(10.0/7.0));
				b = 1.0/3.0*sqrt(5.0-2.0*sqrt(10.0/7.0));
				w1 = (322.0-13.0*sqrt(70.0))/900.0;
				w2 = (322.0+13.0*sqrt(70.0))/900.0;
				*quad_1d_point = {-a, -b, 0, b, a};
				*quad_1d_weight = {w1, w2, 128.0/225.0, w2, w1};
		}
		else if (ind == 16) {
				double a1,a2,a3,a4,a5,a6,a7,a8;
				double w1,w2,w3,w4,w5,w6,w7,w8;
				a1 = 0.0950125098376374401853193;
				a2 = 0.2816035507792589132304605;
				a3 = 0.4580167776572273863424194;
				a4 = 0.6178762444026437484466718;
				a5 = 0.7554044083550030338951012;
				a6 = 0.8656312023878317438804679;
				a7 = 0.9445750230732325760779884;
				a8 = 0.9894009349916499325961542;
				w1 = 0.1894506104550684962853967;
				w2 = 0.1826034150449235888667637;
				w3 = 0.1691565193950025381893121;
				w4 = 0.1495959888165767320815017;
				w5 = 0.1246289712555338720524763;
				w6 = 0.0951585116824927848099251;
				w7 = 0.0622535239386478928628438;
				w8 = 0.0271524594117540948517806;
				*quad_1d_point = {-a8, -a7, -a6, -a5, -a4, -a3, 
						-a2, -a1, a1, a2, a3, a4, a5, a6, a7, a8};
				*quad_1d_weight = {w8, w7, w6, w5, w4, w3, w2, 
						w1, w1, w2, w3, w4, w5, w6, w7, w8};
		}
		else {
				std::cout << "likely an error in legendre... ind != to anything." << endl;
		}
}

// quad underscore is really a terrible name.
void quad_(double a, double b, int N, double* varargin, int arg_size) {
		double c,h;
		c = (a + b)/2.0;
		h = (b - a)/2.0;



		double* x = (double*) calloc(N, sizeof(double));
		double* w = (double*) calloc(N, sizeof(double));

		// h is overwritten here without being used... 
		// these frequency of these little things makes me
		// wonder if the matlab code is correct
		h=(b-a)/(N-1);
		
		int p=2;
		double[p] quad;
		double[p] w_;
		for (int i=0; i<N/2; i++) {
				legendre(p, quad, w_);
				int k = 0;
				//TODO: check these indices 
				for (int j=p*i; j<p*(i+1); j++) {
					x[j] = a+h*((2*i)+quad[k]);
					w[j] = w_[k++];
				}

		//x([(p*(i-1)+1):p*i])= a+ h*((2*i-1) +  quad);
		//w([(p*(i-1)+1):p*i])=w_;
		}
		double* y = new double[N];
		//y = feval(funfcn,x,varargin{:});
		//Q=h*y*w;
		double* Q = new double[N];
		for (int i=0; i<N; i++) {
			//TODO: y[i] = HestonPIntegrand...
			Q[i] = h[i]*y[i]*w[i];
		}
		free(x);
		free(w);
		free(y);
}

