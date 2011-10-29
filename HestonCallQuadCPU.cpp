#include "HestonCallQuadCPU.hpp"
#include <complex>
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#define pi 3.1415926535897932384626433832795

// bit of matlab syntax in here still--won't compile (yet)
// benchmark.m takes 151 seconds for me

inline std::complex<double> Hestf(
        double phi, 
        double kappa, 
        double theta, 
        double sigma,
        double rho, 
        double v0, 
        double r, 
        double T, 
        double s0,
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
    std::complex<double> zI(0, 1);

    double a,x;
    std::complex<double> d,g,C,D,f;
    //TODO: fix this stuff
    a = kappa*theta;
    x = std::log(s0);
    d = std::sqrt(pow(rho*sigma*phi*zI-b,2)-(sigma*sigma)*(2*u*phi*zI-(phi*phi)));
    g = (b-rho*sigma*phi*zI + d)/(b-rho*sigma*phi*zI - d);
    C = r*phi*zI*T + a/(sigma*sigma)*((b- rho*sigma*phi*zI + d)*T 
            - 2.0*std::log((1.0-g*std::exp(d*T))/(1.0-g)));
    D = (b-rho*sigma*phi*zI + d)/(sigma*sigma)*((1.0-std::exp(d*T))/ (1.0-g*std::exp(d*T)));

    f = exp(C + D*v0 + zI*phi*x);
    return f;
}


inline double hestonPIntegrand(
        double phi, 
        double kappa,
        double theta,
        double sigma,
        double rho,
        double v0,
        double r,
        double T,
        double s0,
        double K,
        int type) {

    std::complex<double> zI(0, 1);

    return (std::exp(-zI*phi*std::log(K))
            *Hestf(phi,kappa,theta,sigma,
                rho,v0,r,T,s0,type)/(zI*phi)).real();

}

inline void legendre(
        int ind, 
        double* quad_1d_point, 
        double* quad_1d_weight) {

    double* q = quad_1d_point;
    double* w = quad_1d_weight;

    if (ind == 1) {
        *quad_1d_point = 0;
        *quad_1d_weight = 2;
    }
    else if (ind == 2) {
        //*quad_1d_point = {-1.0/sqrt(3.0), 1.0/sqrt(3.0)};
        q[0]= -1.0/sqrt(3.0);
        q[1]= 1.0/sqrt(3.0);
        //*quad_1d_weight = {1, 1};
        w[0] = 1;
        w[1] = 1;
    }
    else if (ind == 3) {
        //*quad_1d_point = {-sqrt(3.0/5.0), 0, sqrt(3.0/5.0)};
        q[0] = -sqrt(3.0/5.0); 
        q[1] = 0; 
        q[2] = sqrt(3.0/5.0);
        //*quad_1d_weight = {5.0/9.0, 8.0/9.0, 5.0/9.0};
        w[0]= 5.0/9.0; 
        w[1]= 8.0/9.0; 
        w[2]= 5.0/9.0;
    }
    else if (ind == 4) {
        double a,b,w1,w2;
        a = sqrt((3.0+2.0*sqrt(6.0/5.0)/7.0));
        b = sqrt((3.0-2.0*sqrt(6.0/5.0)/7.0));
        w1 = (18.0-sqrt(30.0))/36.0;
        w2 = (18.0+sqrt(30.0))/36.0;
        //*quad_1d_point = {-a, -b, b, a};
        q[0] = -a; 
        q[1] = -b; 
        q[2] = b; 
        q[3] = a;
        //*quad_1d_weight = {w1, w2, w2, w1};
        w[0] = w1;
        w[1] = w2;
        w[2] = w2;
        w[3] = w1;
    }
    else if (ind == 5) {
        double a,b,w1,w2;
        a = 1.0/3.0*sqrt(5.0+2.0*sqrt(10.0/7.0));
        b = 1.0/3.0*sqrt(5.0-2.0*sqrt(10.0/7.0));
        w1 = (322.0-13.0*sqrt(70.0))/900.0;
        w2 = (322.0+13.0*sqrt(70.0))/900.0;
        //*quad_1d_point = {-a, -b, 0, b, a};
        q[0] = -a; 
        q[1] = -b; 
        q[2] = 0; 
        q[3] = b; 
        q[4] = a;
        //*quad_1d_weight = {w1, w2, 128.0/225.0, w2, w1};
        w[0] = w1; 
        w[1] = w2;
        w[2] = 128.0/225.0; 
        w[3] = w2; 
        w[4] = w1;
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
        //*quad_1d_point = {-a8, -a7, -a6, -a5, -a4, -a3, 
        //    -a2, -a1, a1, a2, a3, a4, a5, a6, a7, a8};
        //*quad_1d_weight = {w8, w7, w6, w5, w4, w3, w2, 
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
    else {
        std::cout << "likely an error in legendre... ind != to anything." 
            << std::endl;
        throw;
    }
}

// quad_ is really a terrible function name.
// 0,100,N, kappa, theta,sigma,rho,v0,r,T,s0,K,type
inline double quad_(
        double a, 
        double b, 
        int N, 
        double kappa, 
        double theta,
        double sigma, 
        double rho, 
        double v0, 
        double r, 
        double T,
        double s0, 
        double K, 
        int type) {

    double c,h;
    c = (a + b)/2.0;
    h = (b - a)/2.0;

    double* x = new double[N];
    double* w = new double[N];
    //double* x = (double*) calloc(N, sizeof(double));
    //double* w = (double*) calloc(N, sizeof(double));

    // h is overwritten here without being used... 
    // these frequency of these little things makes me
    // wonder if the matlab code is correct
    h=(b-a)/(N-1);

    const int p=2; // fixed in quad_
    //double[p] quad;
    //double[p] w_;
    double* quad = new double[p];
    double* w_ = new double[p];

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
    //double* Q = new double[N];
    double Q = 0;

    for (int i=0; i<N; i++) {
        y[i] = hestonPIntegrand(x[i], kappa, theta, 
                sigma, rho, v0, r, T, s0, K, type);
        //Q[i] = h*y[i]*w[i];
        Q += y[i]*w[i];
    }
    Q *= h;
    //free(x);
    //free(w);
    delete[] quad;
    delete[] w_;
    delete[] x;
    delete[] w;
    delete[] y;
    return Q;
}

inline double HestonP(
        double kappa, 
        double theta, 
        double sigma, 
        double rho,
        double v0, 
        double r, 
        double T, 
        double s0, 
        double K, 
        int type, 
        int N) {

    // previously, the arg @HestonPIntegrand was passed to quadl,
    // which specified the quadrature function.
    // HestonPIntegrand is now coded into quadl as the function
    // since it is the only function used for quadl in our code.

    //TODO: fix pi; return type of quad_
    return 0.5 + 1/pi*
        quad_(0,100,N, kappa, theta,sigma,rho,v0,r,T,s0,K,type);
}

double HestonCallQuadCPU(
        double kappa,
        double theta,
        double sigma,
        double rho,
        double v0,
        double r,
        double T, 
        double s0,
        int K,
        int N) {

    //const int N = 0; //constant in the MATLAB code... (and here)

    return s0*HestonP(kappa,theta,sigma,rho,v0,r,T,s0,K,1,N) - 
        K*std::exp(-r*T)*HestonP(kappa,theta,sigma,rho,v0,r,T,s0,K,2,N);

}

