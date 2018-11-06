__global__ void pre_mul_kernel(int n, double *a, double *ct) {
	const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
	double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki;
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
		a[j] = ajr;
		a[1 + j] = aji;
		a[nminusj] = akr;
		a[1 + nminusj] = aki;
	}
}

void cudaAcc_pre_mul(int threads, int n, double *a, double *ct) {
	pre_mul_kernel<<<n / (4 * threads), threads>>>(n, a, ct);
}
