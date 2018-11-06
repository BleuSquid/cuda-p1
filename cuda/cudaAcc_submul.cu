__global__ void sub_mul_kernel(double *g_out, double *a, double *b1, double *b2, double *ct, int n) {
	const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
	double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki, bjr, bji, bkr, bki;
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
		xr = ajr - akr;
		xi = aji + aki;
		yr = wkr * xr - wki * xi;
		yi = wkr * xi + wki * xr;
		ajr -= yr;
		aji -= yi;
		akr += yr;
		aki -= yi;

		bjr = b1[j] - b2[j];
		bji = b1[1 + j] - b2[1 + j];
		bkr = b1[nminusj] - b2[nminusj];
		bki = b1[1 + nminusj] - b2[1 + nminusj];

		new_aji = ajr * bji + bjr * aji;
		new_ajr = ajr * bjr - aji * bji;
		new_aki = akr * bki + bkr * aki;
		new_akr = akr * bkr - aki * bki;

		xr = new_ajr - new_akr;
		xi = new_aji + new_aki;
		yr = wkr * xr + wki * xi;
		yi = wkr * xi - wki * xr;
		g_out[j] = new_ajr - yr;
		g_out[1 + j] = yi - new_aji;
		g_out[nminusj] = new_akr + yr;
		g_out[1 + nminusj] = yi - new_aki;
	} else {
		xr = a[0];
		xi = a[1];
		yr = b1[0] - b2[0];
		yi = b1[1] - b2[1];
		g_out[0] = xr * yr + xi * yi;
		g_out[1] = -xr * yi - xi * yr;
		xr = a[0 + m];
		xi = a[1 + m];
		yr = b1[0 + m] - b2[0 + m];
		yi = b1[1 + m] - b2[1 + m];
		g_out[1 + m] = -xr * yi - xi * yr;
		g_out[0 + m] = xr * yr - xi * yi;
	}
}

void cudaAcc_sub_mul(int threads, int n, double *g_out, double *a, double *b1, double *b2, double *ct) {
	sub_mul_kernel<<<n / (4 * threads), threads>>>(g_out, a, b1, b2, ct, n);
}
