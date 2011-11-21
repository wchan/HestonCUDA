#include "HestonCallQuadGPU.hpp"
#include <complex>
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include "HestonCUDA.hpp"
#define pi 3.1415926535897932384626433832795

// NVIDIA CUDA Headers
#include <cuda.h>
#include <cuComplex.h>

// NVIDIA Thrust Headers (http://developer.nvidia.com/Thrust)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

struct HestonCallQuadGPU_functor1 {
	const HestonCUDAPrecision dKappa;
	const HestonCUDAPrecision dTheta;
	const HestonCUDAPrecision dSigma;
	const HestonCUDAPrecision dRho;
	const HestonCUDAPrecision dV0;
	const HestonCUDAPrecision dR;
	const HestonCUDAPrecision dT;
	const HestonCUDAPrecision dS0;
	const HestonCUDAPrecision dStrike;
	const int type;
	const int p;
	const HestonCUDAPrecision a;
	const HestonCUDAPrecision h;

	HestonCallQuadGPU_functor1(
			HestonCUDAPrecision dKappa,   // rate of reversion
			HestonCUDAPrecision dTheta,   // int run variance
			HestonCUDAPrecision dSigma,   // vol of vol
			HestonCUDAPrecision dRho,     // correlation
			HestonCUDAPrecision dV0,      // initial variance
			HestonCUDAPrecision dR,       // instantaneous short rate
			HestonCUDAPrecision dT,       // time till maturity
			HestonCUDAPrecision dS0,      // initial asset price
			HestonCUDAPrecision dStrike,
			int p,
			HestonCUDAPrecision a,
			HestonCUDAPrecision h,
			int type

			) : dKappa(dKappa), dTheta(dTheta), dSigma(dSigma), dRho(dRho),
			dV0(dV0), dR(dR), dT(dT), dS0(dS0), dStrike(dStrike), p(p),
			a(a), h(h), type(type) {}

	__host__ __device__
	HestonCUDAPrecision operator()(const int& i, const HestonCUDAPrecision& quad) const {
		return a+h*((2*(i/p)+1)+quad); // return x
	}
};

//    for (int i=0; i<N; i++) {
//		** func1 **
//		x = a+h*((2*(i/p)+1)+quad[i%p]);
//		** func2 **
//		w = w_[i%p];
//		y = HestonP_GPUIntegrand(x, kappa, theta,
//				sigma, rho, v0, r, T, s0, K, type);
//		Q += y*w;
//    }

	struct HestonCallQuadGPU_functor2 {
		const HestonCUDAPrecision dKappa;
		const HestonCUDAPrecision dTheta;
		const HestonCUDAPrecision dSigma;
		const HestonCUDAPrecision dRho;
		const HestonCUDAPrecision dV0;
		const HestonCUDAPrecision dR;
		const HestonCUDAPrecision dT;
		const HestonCUDAPrecision dS0;
		const HestonCUDAPrecision dStrike;
		const int type;
		const int p;
		const HestonCUDAPrecision a;
		const HestonCUDAPrecision h;

		HestonCallQuadGPU_functor2(
				HestonCUDAPrecision dKappa,   // rate of reversion
				HestonCUDAPrecision dTheta,   // int run variance
				HestonCUDAPrecision dSigma,   // vol of vol
				HestonCUDAPrecision dRho,     // correlation
				HestonCUDAPrecision dV0,      // initial variance
				HestonCUDAPrecision dR,       // instantaneous short rate
				HestonCUDAPrecision dT,       // time till maturity
				HestonCUDAPrecision dS0,      // initial asset price
				HestonCUDAPrecision dStrike,
				int p,
				HestonCUDAPrecision a,
				HestonCUDAPrecision h,
				int type

				) : dKappa(dKappa), dTheta(dTheta), dSigma(dSigma), dRho(dRho),
					dV0(dV0), dR(dR), dT(dT), dS0(dS0), dStrike(dStrike), p(p),
					a(a), h(h), type(type) {}

		__host__ //__device__
		HestonCUDAPrecision operator()(const HestonCUDAPrecision& x, const HestonCUDAPrecision& w) {
			HestonCUDAPrecision y,Q;
			y = HestonP_GPUIntegrand(x, dKappa, dTheta,
					dSigma, dRho, dV0, dR, dT, dS0, dStrike, type);

//			std::cout<<x<<" " << dKappa << " " << dTheta
//					<< " " << dSigma << " " << dRho <<" "
//				<< dV0 << " " << dR << " " << dT << " "
//					<< dS0 << " " << dStrike << " " << type <<
//					y << std::endl;

			Q = y*w;
			//std::cout << "x: " << x << "\ty: " << y << std::endl;
			return Q;
		}

	__host__ __device__ inline
	HestonCUDAPrecisionComplex Hestf_GPU(
	        HestonCUDAPrecision phi,
	        HestonCUDAPrecision kappa,
	        HestonCUDAPrecision theta,
	        HestonCUDAPrecision sigma,
	        HestonCUDAPrecision rho,
	        HestonCUDAPrecision v0,
	        HestonCUDAPrecision r,
	        HestonCUDAPrecision T,
	        HestonCUDAPrecision s0,
	        int type) {

	    HestonCUDAPrecision u, b;

	    if (type == 1) {
	        u = 0.5;
	        b = kappa - (rho*sigma);
	    }
	    else {
	        u = -0.5;
	        b = kappa;
	    }
	    HestonCUDAPrecisionComplex zI = make_complex(0, 1);

	    HestonCUDAPrecision a,x;
	    HestonCUDAPrecisionComplex d,g,C,D,f;

		a = kappa*theta;
	    x = log(s0);
	    d = sqrt(pow(rho*sigma*phi*zI-b,2)-(sigma*sigma)*(2*u*phi*zI-(phi*phi)));

	    g = (b-rho*sigma*phi*zI + d)/(b-rho*sigma*phi*zI - d);
	    C = r*phi*zI*T + a/(sigma*sigma)*((b- rho*sigma*phi*zI + d)*T
	            - 2.0*log((1.0-g*exp(d*T))/(1.0-g)));

	    D = (b-rho*sigma*phi*zI + d)/(sigma*sigma)*((1.0-exp(d*T))/ (1.0-g*exp(d*T)));
	    //std::cout << D.x << " " << D.y << std::endl;

	    f = exp(C + D*v0 + zI*phi*x);

	    //std::cout << (C + D*v0 + zI*phi*x).x << " " << (C + D*v0 + zI*phi*x).y << std::endl;

	    return f;
	}


	__host__ __device__ inline
	HestonCUDAPrecision HestonP_GPUIntegrand(
	        HestonCUDAPrecision phi,
	        HestonCUDAPrecision kappa,
	        HestonCUDAPrecision theta,
	        HestonCUDAPrecision sigma,
	        HestonCUDAPrecision rho,
	        HestonCUDAPrecision v0,
	        HestonCUDAPrecision r,
	        HestonCUDAPrecision T,
	        HestonCUDAPrecision s0,
	        HestonCUDAPrecision K,
	        int type) {

	    HestonCUDAPrecisionComplex zI = make_complex(0, 1);

		HestonCUDAPrecisionComplex p1,p2,p3;
		p1 = exp(-zI*phi*log(K));
		p2 = Hestf_GPU(phi,kappa,theta,sigma,rho,v0,r,T,s0,type);
		p3 = zI*phi;
	    return (p1*p2/p3).x;

	}

};

__host__ inline void legendre_GPU(
        int ind, 
        HestonCUDAPrecision* q,
        HestonCUDAPrecision* w) {

    if (ind == 1) {
        q[0] = 0;
        w[0] = 2;
    }
    else if (ind == 2) {
        //*quad_GPU1d_point = {-1.0/sqrt(3.0), 1.0/sqrt(3.0)};
        q[0]= -1.0/sqrt(3.0);
        q[1]= 1.0/sqrt(3.0);
        //*quad_GPU1d_weight = {1, 1};
        w[0] = 1;
        w[1] = 1;
    }
    else if (ind == 3) {
        //*quad_GPU1d_point = {-sqrt(3.0/5.0), 0, sqrt(3.0/5.0)};
        q[0] = -sqrt(3.0/5.0);
        q[1] = 0;
        q[2] = sqrt(3.0/5.0);
        //*quad_GPU1d_weight = {5.0/9.0, 8.0/9.0, 5.0/9.0};
        w[0]= 5.0/9.0;
        w[1]= 8.0/9.0;
        w[2]= 5.0/9.0;
    }
    else if (ind == 4) {
        HestonCUDAPrecision a,b,w1,w2;
        a = sqrt((3.0+2.0*sqrt(6.0/5.0)/7.0));
        b = sqrt((3.0-2.0*sqrt(6.0/5.0)/7.0));
        w1 = (18.0-sqrt(30.0))/36.0;
        w2 = (18.0+sqrt(30.0))/36.0;
        //*quad_GPU1d_point = {-a, -b, b, a};
        q[0] = -a;
        q[1] = -b;
        q[2] = b;
        q[3] = a;
        //*quad_GPU1d_weight = {w1, w2, w2, w1};
        w[0] = w1;
        w[1] = w2;
        w[2] = w2;
        w[3] = w1;
    }
    else if (ind == 5) {
        HestonCUDAPrecision a,b,w1,w2;
        a = 1.0/3.0*sqrt(5.0+2.0*sqrt(10.0/7.0));
        b = 1.0/3.0*sqrt(5.0-2.0*sqrt(10.0/7.0));
        w1 = (322.0-13.0*sqrt(70.0))/900.0;
        w2 = (322.0+13.0*sqrt(70.0))/900.0;
        //*quad_GPU1d_point = {-a, -b, 0, b, a};
        q[0] = -a;
        q[1] = -b;
        q[2] = 0;
        q[3] = b;
        q[4] = a;
        //*quad_GPU1d_weight = {w1, w2, 128.0/225.0, w2, w1};
        w[0] = w1;
        w[1] = w2;
        w[2] = 128.0/225.0;
        w[3] = w2;
        w[4] = w1;
    }
    else if (ind == 16) {
        HestonCUDAPrecision a1,a2,a3,a4,a5,a6,a7,a8;
        HestonCUDAPrecision w1,w2,w3,w4,w5,w6,w7,w8;
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
        //*quad_GPU1d_point = {-a8, -a7, -a6, -a5, -a4, -a3,
        //    -a2, -a1, a1, a2, a3, a4, a5, a6, a7, a8};
        //*quad_GPU1d_weight = {w8, w7, w6, w5, w4, w3, w2,
        //    w1, w1, w2, w3, w4, w5, w6, w7, w8};

        q[0] = -a8;
        q[1] = -a7;
        q[2] = -a6;
        q[3] = -a5;
        q[4] = -a4;
        q[5] = -a3;
        q[6] = -a2;
        q[7] = -a1;
        q[8] = a1;
        q[9] = a2;
        q[10] = a3;
        q[11] = a4;
        q[12] = a5;
        q[13] = a6;
        q[14] = a7;
        q[15] = a8;

        w[0] = w8;
        w[1] = w7;
        w[2] = w6;
        w[3] = w5;
        w[4] = w4;
        w[5] = w3;
        w[6] = w2;
        w[7] = w1;
        w[8] = w1;
        w[9] = w2;
        w[10] = w3;
        w[11] = w4;
        w[12] = w5;
        w[13] = w6;
        w[14] = w7;
        w[15] = w8;


    }
    else
        throw "likely an error in legendre_GPU... ind != to anything." ;
}

__host__ inline HestonCUDAPrecision quad_GPU(
        HestonCUDAPrecision a, 
        HestonCUDAPrecision b, 
        long N, 
        HestonCUDAPrecision kappa, 
        HestonCUDAPrecision theta,
        HestonCUDAPrecision sigma, 
        HestonCUDAPrecision rho, 
        HestonCUDAPrecision v0, 
        HestonCUDAPrecision r, 
        HestonCUDAPrecision T,
        HestonCUDAPrecision s0, 
        HestonCUDAPrecision K, 
        int type) {

    HestonCUDAPrecision h; // var c isn't ever used
    //c = (a + b)/2.0;
    h = (b - a)/2.0;

    h=(b-a)/(N-1);

    // tested with p=2 only
    const int p=2; // fixed in quad_GPU


    HestonCUDAPrecision* w_ = new HestonCUDAPrecision[p];
    HestonCUDAPrecision* quad = new HestonCUDAPrecision[p];

    legendre_GPU(p, quad, w_);

    thrust::host_vector<HestonCUDAPrecision> dev_quad(&quad[0], &quad[p]);
    thrust::host_vector<HestonCUDAPrecision> dev_w(&w_[0], &w_[p]);

    // TODO: using int not long...
    thrust::counting_iterator<int> counter(0);
    thrust::counting_iterator<int> counter_last = counter+N;
    thrust::constant_iterator<int> p_it(p);

    thrust::host_vector<int> indicies(N);
    thrust::transform(counter, counter_last, p_it, indicies.begin(), thrust::modulus<int>());

    thrust::host_vector<HestonCUDAPrecision> dev_full_quad(N);
    thrust::host_vector<HestonCUDAPrecision> dev_full_w(N);



    // get w_ and quad vecs set up
    //TODO: use fancy iterators here and for indicies
    thrust::copy(
    		thrust::make_permutation_iterator(dev_quad.begin(),indicies.begin()),
    		thrust::make_permutation_iterator(dev_quad.begin(),indicies.end()),
    		dev_full_quad.begin()
    );
    thrust::copy(
        		thrust::make_permutation_iterator(dev_w.begin(),indicies.begin()),
        		thrust::make_permutation_iterator(dev_w.begin(),indicies.end()),
        		dev_full_w.begin()
    );
    //debugging
//    thrust::copy(dev_full_w.begin(),dev_full_w.end(),std::ostream_iterator<double>(std::cout," "));
//    thrust::copy(dev_w.begin(),dev_w.end(),std::ostream_iterator<double>(std::cout,"\n\n"));

    thrust::host_vector<HestonCUDAPrecision> dev_Q(N);
    HestonCallQuadGPU_functor1 func1(kappa,theta,sigma,rho,v0,r,T,s0,K,p,a,h,type); // calculates x
    HestonCallQuadGPU_functor2 func2(kappa,theta,sigma,rho,v0,r,T,s0,K,p,a,h,type); // calculates Q

    thrust::transform(counter, counter_last,
    		dev_full_quad.begin(),
    		dev_Q.begin(), func1);


    thrust::transform(dev_Q.begin(), dev_Q.end(),
        		dev_full_w.begin(),
        		dev_Q.begin(), func2);

    //thrust::copy(dev_Q.begin(),dev_Q.end(),std::ostream_iterator<double>(std::cout,"\n"));


    HestonCUDAPrecision Q = thrust::reduce(dev_Q.begin(), dev_Q.end());
//    for (int i=0; i<N; i++) {
//		x = a+h*((2*(i/p)+1)+quad[i%p]);
//		w = w_[i%p];
//		y = HestonP_GPUIntegrand(x, kappa, theta,
//				sigma, rho, v0, r, T, s0, K, type);
//		Q += y*w;
//    }
    Q *= h;
    delete[] quad;
    delete[] w_;
//    delete[] x;
//    delete[] w;
    //delete[] y;
    return Q;
}

__host__ inline HestonCUDAPrecision HestonP_GPU(
        HestonCUDAPrecision kappa, 
        HestonCUDAPrecision theta, 
        HestonCUDAPrecision sigma, 
        HestonCUDAPrecision rho,
        HestonCUDAPrecision v0, 
        HestonCUDAPrecision r, 
        HestonCUDAPrecision T, 
        HestonCUDAPrecision s0, 
        HestonCUDAPrecision K, 
        int type, 
        long N) {


    return 0.5 + 1/pi*
        quad_GPU(0,100,N, kappa, theta,sigma,rho,v0,r,T,s0,K,type);
}

__host__ HestonCUDAPrecision HestonCallQuadGPU(
		HestonCUDAPrecision kappa,   // rate of reversion
		HestonCUDAPrecision theta,   // int run variance
		HestonCUDAPrecision sigma,   // vol of vol
		HestonCUDAPrecision rho,     // correlation
		HestonCUDAPrecision v0,      // initial variance
		HestonCUDAPrecision r,       // instantaneous short rate
		HestonCUDAPrecision T,       // time till maturity
		HestonCUDAPrecision s0,      // initial asset price
		HestonCUDAPrecision K,
		long   N) {

	return s0*HestonP_GPU(kappa,theta,sigma,rho,v0,r,T,s0,K,1,N) -
        K*exp(-r*T)*HestonP_GPU(kappa,theta,sigma,rho,v0,r,T,s0,K,2,N);
}

