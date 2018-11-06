__global__ void square1_kernel(int n, double *b, double *a, double *ct) {
	const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
	double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki;
	double new_ajr, new_aji, new_akr, new_aki;
	const int m = n >> 1;
	const int nc = n >> 2;
	const int j = j2 << 1;

	if (j2) {
		int nminusj = n - j;
		wkr = 0.5 - ct[nc - j2];
		wki = ct[j2];
		ajr = a[j];
		aji = a[1 + j];
		akr = a[nminusj];
		aki = a[1 + nminusj];

		new_aji = 2.0 * ajr * aji;
		new_ajr = (ajr - aji) * (ajr + aji);
		new_aki = 2.0 * akr * aki;
		new_akr = (akr - aki) * (akr + aki);

		xr = new_ajr - new_akr;
		xi = new_aji + new_aki;
		yr = wkr * xr + wki * xi;
		yi = wkr * xi - wki * xr;

		b[j] = new_ajr - yr;
		b[1 + j] = yi - new_aji;
		b[nminusj] = new_akr + yr;
		b[1 + nminusj] = yi - new_aki;
	} else {
		xr = a[0];
		xi = a[1];
		b[0] = xr * xr + xi * xi;
		b[1] = -xr * xi - xi * xr;
		xr = a[0 + m];
		xi = a[1 + m];
		b[1 + m] = -xr * xi - xi * xr;
		b[0 + m] = xr * xr - xi * xi;
	}
}

void cudaAcc_square1(int threads, int n, double *b, double *a, double *ct) {
	square1_kernel<<< n / (4 * threads), threads >>>(n, b, a, ct);
}
