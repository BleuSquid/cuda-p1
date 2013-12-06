#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define D 25
#define C 1000

long double rho_pre[C][D];

void rho_init() {
	int i;
	int j;
	long double minus1overu;
	long double r;
	
	long double half[D];
	long double integ[D];

	half[0] = 0.5;

	for (j = 1;j < D; ++j)
		half[j] = half[j - 1] * 0.5;
	
	for (j = 0;j < D; ++j)
		integ[j] = half[j] / (long double)(j + 1.0);
	
	rho_pre[0][0] = 1;
	rho_pre[1][0] = 1;

	for (j = 1;j < D; ++j) {
		rho_pre[0][j] = 0;
		rho_pre[1][j] = 0;
	}
	
	for (i = 2;i < C; ++i) {
		minus1overu = -2.0 / (long double) i;
		
		r = 0.0;
		for (j = 0;j < D; ++j)
			r += rho_pre[i - 2][j] * integ[j] + rho_pre[i - 1][j] * integ[j];

		r *= -minus1overu;

		rho_pre[i][0] = r;
		rho_pre[i][1] = minus1overu * rho_pre[i - 2][0];

		for (j = 2;j < D;++j) {
			rho_pre[i][j] = (minus1overu / (long double) j) * (rho_pre[i - 2][j - 1] + rho_pre[i][j - 1] * ((long double) j - 1.0));
		}
	}
}

inline double rho(double u) {
	int i;
	int j;
	long double eps;
	long double result;
	long double u2 = 1.0 / (long double) u;
	
	if (u2 > 1000.0)
		return 0;
	
	if (u2 < 0)
		return 1;
	
	i = (int) floor(2.0 * u2);
	
	if (i >= C)
		return 0;
	if (i < 0)
		return 0;
	
	eps = u2 - (long double) i / 2.0;

	result = 0.0;
	for (j = D - 1;j >= 0;--j)
		result = result * eps + rho_pre[i][j];

	return result;
}
/*
int main(int argc,char **argv) {
	double u;
	long double rhou;

	rho_init();
	
	while (scanf("%lf",&u) == 1) {
		rhou = rho(u);
		
		printf("%10.10Lg\n",rhou);
	}
	
	exit(0);
}*/
