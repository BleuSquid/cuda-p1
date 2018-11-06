#include "parse.h"
#include "CUDAPm1.h"
#include "cuda/cuda_safecalls.h"
#include "cuda/cuda_functions.h"

void init_threads(int n) {
	FILE *threads;
	char buf[132];
	char threadfile[256];
	int no_file = 0, no_entry = 1;
	int th1 = 0, th2 = 0, th3 = 0;
	int temp_n;

	char devname[256];
	remove_spaces(devname, dev.name);

	sprintf(threadfile, "%s_threads.txt", devname);
	threads = fopen(threadfile, "r");
	if (threads) {
		while (fgets(buf, 132, threads) != NULL) {
			sscanf(buf, "%d %d %d %d", &temp_n, &th1, &th2, &th3);
			if (n == temp_n * 1024) {
				threads1 = th1;
				threads2 = th2;
				threads3 = th3;
				no_entry = 0;
			}
		}
	} else
		no_file = 1;
	if (no_file || no_entry) {
		if (no_file)
			printf("No %s file found. Using default thread sizes.\n", threadfile);
		else if (no_entry)
			printf("No entry for fft = %dk found. Using default thread sizes.\n", n / 1024);
		printf("For optimal thread selection, please run\n");
		printf("./CUDAPm1 -cufftbench %d %d r\n", n / 1024, n / 1024);
		printf("for some small r, 0 < r < 6 e.g.\n");
		fflush (NULL);
	}
	return;
}

float lucas_square(/*double *x,*/int q, int n, int iter, int last, float* maxerr, int error_flag, int bit, int stage, int chkpt) {
	float terr = 0.0;

	if (iter < 100 && iter % 10 == 0) {
		cutilSafeCall(cudaMemcpy(&terr, g_err, sizeof(float), cudaMemcpyDeviceToHost));
		if (terr > *maxerr)
			*maxerr = terr;
	}
	cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
	cudaAcc_square(threads2, n, g_x, g_ct);
	cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));

	if (chkpt) {
		if (!bit) {
			cudaAcc_norm1a(1, threads1, n, g_x, g_datai, g_xint, g_ttmp, g_carryi, g_err, *maxerr);
			cudaAcc_norm2a(1, threads1, threads3, g_x, g_xint, n, g_datai, g_carryi, g_ttp1);
		} else {
			cudaAcc_norm1b(1, threads1, n, g_x, g_datal, g_xint, g_ttmp, g_carryl, g_err, *maxerr);
			cudaAcc_norm2b(1, threads1, threads3, g_x, g_xint, n, g_datal, g_carryl, g_ttp1);
		}
	} else {
		if (!bit) {
			cudaAcc_norm1a(0, threads1, n, g_x, g_datai, g_xint, g_ttmp, g_carryi, g_err, *maxerr);
			cudaAcc_norm2a(0, threads1, threads3, g_x, g_xint, n, g_datai, g_carryi, g_ttp1);
		} else {
			cudaAcc_norm1b(0, threads1, n, g_x, g_datal, g_xint, g_ttmp, g_carryl, g_err, *maxerr);
			cudaAcc_norm2b(0, threads1, threads3, g_x, g_xint, n, g_datal, g_carryl, g_ttp1);
		}
	}

	if (error_flag) {
		cutilSafeCall(cudaMemcpy(&terr, g_err, sizeof(float), cudaMemcpyDeviceToHost));
		if (terr > *maxerr)
			*maxerr = terr;
	} else if (polite_f && (iter % polite) == 0) {
		cutilSafeThreadSync();
	}
	return (terr);
}

int choose_fft_length(int q, int *index) {
	/* In order to increase length if an exponent has a round off issue, we use an
	 extra paramter that we can adjust on the fly. In check(), index starts as -1,
	 the default. In that case, choose from the table. If index >= 0, we must assume
	 it's an override index and return the corresponding length. If index > table-count,
	 then we assume it's a manual fftlen and return the proper index. */

	if (0 < *index && *index < fft_count)
		return 1024 * multipliers[*index];
	else if (*index >= fft_count || q == 0) { /* override with manual fftlen passed as arg; set pointer to largest index <= fftlen */
		int len, i;
		for (i = fft_count - 1; i >= 0; i--) {
			len = 1024 * multipliers[i];
			if (len <= *index) {
				*index = i;
				return len; /* not really necessary, but now we could decide to override fftlen with this value */
			}
		}
	} else {  // *index < 0, not override, choose length and set pointer to proper index
		int i;
		int estimate = ceil(1.01 * 0.0000358738168878758 * exp(1.0219834608 * log((double) q)));

		for (i = 0; i < fft_count; i++) {
			if (multipliers[i] >= estimate) {
				*index = i;
#ifdef EBUG
				printf("Index %d\n", *index);
#endif
				return multipliers[i] * 1024;
			}
		}
	}
	return 0;
}

/* -------- initializing routines -------- */
void makect(int nc, double *c) {
	int j;
	double d = (double) (nc << 1);

	for (j = 0; j < nc; j++)
		c[j] = 0.5 * cospi(j / d);
}

void alloc_gpu_mem(int n) {
	cufftSafeCall(cufftPlan1d(&plan, n / 2, CUFFT_Z2Z, 1));
	cutilSafeCall(cudaMalloc((void **) &g_x, sizeof(double) * n));
	cutilSafeCall(cudaMalloc((void **) &g_ct, sizeof(double) * n / 4));
	cutilSafeCall(cudaMalloc((void **) &g_xint, sizeof(int) * 2 * n));
	cutilSafeCall(cudaMalloc((void **) &g_err, sizeof(float)));
	cutilSafeCall(cudaMalloc((void **) &g_ttmp, sizeof(double) * n));
	cutilSafeCall(cudaMalloc((void **) &g_ttp1, sizeof(double) * 2 * n / threads1));
	cutilSafeCall(cudaMalloc((void **) &g_datai, sizeof(int) * 2 * n / threads1));
	cutilSafeCall(cudaMalloc((void **) &g_datal, sizeof(long long int) * 2 * n / threads1));
	cutilSafeCall(cudaMemset(g_err, 0, sizeof(float)));
	cutilSafeCall(cudaMalloc((void **) &g_carryl, sizeof(long long int) * n / threads1));
	cutilSafeCall(cudaMalloc((void **) &g_carryi, sizeof(int) * n / threads1));
}

void write_gpu_data(int q, int n) {
	double *s_ttmp, *s_ttp1, *s_ct;
	int i, j, qn = q / n, b = q % n;
	int a, c, bj;
	double *h_ttp_inc;
	int *h_qn;

	s_ct = (double *) malloc(sizeof(double) * (n / 4));
	s_ttmp = (double *) malloc(sizeof(double) * (n));
	s_ttp1 = (double *) malloc(sizeof(double) * 2 * (n / threads1));
	size = (char *) malloc(sizeof(char) * n);
	h_ttp_inc = (double *) malloc(sizeof(double) * 2);
	h_qn = (int *) malloc(sizeof(int) * 2);

	c = n - b;
	bj = 0;
	for (j = 1; j < n; j++) {
		bj += b;
		bj %= n;
		a = bj - n;
		if (j % 2 == 0)
			s_ttmp[j] = exp2(a / (double) n) * 2.0 / n;
		else
			s_ttmp[j] = exp2(-a / (double) n);
		size[j] = (bj >= c);
		if (size[j])
			s_ttmp[j] = -s_ttmp[j];
	}
	size[0] = 1;
	s_ttmp[0] = -2.0 / n;
	size[n - 1] = 0;
	s_ttmp[n - 1] = -s_ttmp[n - 1];

	for (i = 0, j = 0; i < n; i += 2 * threads1) {
		s_ttp1[j] = abs(s_ttmp[i + 1]);
		if (size[i])
			s_ttp1[j] = -s_ttp1[j];
		j++;
	}

	makect(n / 4, s_ct);

	h_ttp_inc[0] = -exp2((b - n) / (double) n);
	h_ttp_inc[1] = -exp2(b / (double) n);
	set_ttp_inc(h_ttp_inc);
	h_qn[0] = qn;
	h_qn[1] = n;
	set_qn(h_qn);

	cutilSafeCall(cudaMemcpy(g_ttmp, s_ttmp, sizeof(double) * n, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(g_ttp1, s_ttp1, sizeof(double) * 2 * n / threads1, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(g_ct, s_ct, sizeof(double) * (n / 4), cudaMemcpyHostToDevice));

	free((char *) s_ct);
	free((char *) s_ttmp);
	free((char *) s_ttp1);
	free((char *) h_ttp_inc);
	free((char *) h_qn);
}

void init_x_int(int *x_int, unsigned *x_packed, int q, int n, int *stage) {
	int j;

	if (*stage == 0) {
		*stage = 1;
		for (j = 0; j < n; j++)
			x_int[j] = 0;
		x_int[0] = 1;
		if (x_packed) {
			for (j = 0; j < (q + 31) / 32; j++)
				x_packed[j] = 0;
			x_packed[0] = 1;
		}
	}
	cudaMemcpy(g_xint, x_int, sizeof(int) * n, cudaMemcpyHostToDevice);
}

int* init_lucas_packed_int(unsigned * x_packed, int q, int *n, int *j, int *stage, int *total_time) {
	int *x;
	int new_n, old_n;
	int end = (q + 31) / 32;
	int new_test = 0;

	*n = x_packed[end + 1];
	if (*n == 0)
		new_test = 1;
	*j = x_packed[end + 2];
	*stage = x_packed[end + 3];
	if (total_time)
		*total_time = x_packed[end + 4];

	old_n = fftlen;
	if (fftlen == 0)
		fftlen = *n;
	new_n = choose_fft_length(q, &fftlen);
	if (old_n > fft_count)
		*n = old_n;
	else if (new_test || old_n)
		*n = new_n;
	init_threads(*n);
	printf("Using threads: norm1 %d, mult %d, norm2 %d.\n", threads1, threads2, threads3);
	if ((*n / (2 * threads1)) > dev.maxGridSize[0]) {
		fprintf(stderr, "over specifications Grid = %d\n", (int) *n / (2 * threads1));
		fprintf(stderr, "try increasing norm1 threads (%d) or decreasing FFT length (%dK)\n\n", threads1, *n / 1024);
		return NULL;
	}
	if ((*n / (4 * threads2)) > dev.maxGridSize[0]) {
		fprintf(stderr, "over specifications Grid = %d\n", (int) *n / (4 * threads2));
		fprintf(stderr, "try increasing mult threads (%d) or decreasing FFT length (%dK)\n\n", threads2, *n / 1024);
		return NULL;
	}
	if ((*n % (2 * threads1)) != 0) {
		fprintf(stderr, "fft length %d must be divisible by 2 * norm1 threads %d\n", *n, threads1);
		return NULL;
	}
	if ((*n % (4 * threads2)) != 0) {
		fprintf(stderr, "fft length %d must be divisible by 4 * mult threads %d\n", *n, threads2);
		return NULL;
	}
	if (q < *n * 0.8 * log((double) *n)) {
		fprintf(stderr, "The fft length %dK is too large for the exponent %d. Restart with smaller fft.\n", *n / 1024, q);
		return NULL;
	}
	x = (int *) malloc(sizeof(int) * *n);
	alloc_gpu_mem(*n);
	write_gpu_data(q, *n);
	if (!new_test) {
		unpack_bits_int(x, x_packed, q, *n);
		balance_digits_int(x, q, *n);
	}
	init_x_int(x, x_packed, q, *n, stage);
	cudaAcc_apply_weights(*n, threads1, g_x, g_xint, g_ttmp);
	return x;
}
