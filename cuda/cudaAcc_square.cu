#include "cuda_functions.h"

__global__ void square_kernel(int n, double *a, double *ct) {
	const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
	cuDoubleComplex wk, x, y, aj, ak;
	const int m = n >> 1;
	const int nc = n >> 2;
	const int j = j2 << 1;

	int index[2];

	if (j2) {
		index[0] = j;
		index[1] = n - j;

		wk.x = 0.5 - ct[nc - j2];
		wk.y = ct[j2];
		aj.x = a[index[0]];
		aj.y = a[index[0] + 1];
		ak.x = a[index[1]];
		ak.y = a[index[1] + 1];

		x.x = aj.x - ak.x;
		x.y = aj.y + ak.y;

		y = cuCmul(wk,x);

		aj.x -= y.x;
		aj.y -= y.y;
		ak.x += y.x;
		ak.y -= y.y;

		x = cuCsqr(aj);
		y = cuCsqr(ak);

		aj.x = x.x - y.x;
		aj.y = x.y + y.y;
		ak.x = wk.x * aj.x + wk.y * aj.y;
		ak.y = wk.x * aj.y - wk.y * aj.x;

		aj.x = x.x - ak.x;
		aj.y = ak.y - x.y;
		ak.x = y.x + ak.x;
		ak.y = ak.y - y.y;

	} else {

		index[0] = 0;
		index[1] = m;

		x.x = a[index[0]];
		x.y = a[index[0] + 1];

		aj.x = x.x * x.x + x.y * x.y;
		aj.y = -2 * x.x * x.y;

		y.x = a[index[1]];
		y.y = -a[index[1] + 1];
		ak = cuCsqr(y);
	}
	a[index[0]] = aj.x;
	a[index[0] + 1] = aj.y;
	a[index[1]] = ak.x;
	a[index[1] + 1] = ak.y;
}

void cudaAcc_square(int threads, int n, double *a, double *ct) {
	square_kernel<<< n / (4 * threads), threads >>>(n, a, ct);
}
