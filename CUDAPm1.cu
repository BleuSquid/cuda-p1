char program[] = "CUDAPm1 v0.21";
/* CUDAPM1
   created by owftheevil

   derived from CUDALucas.c
   Shoichiro Yamada Oct. 2010

 This is an adaptation of Richard Crandall lucdwt.c, John Sweeney MacLucasUNIX.c,
 and Guillermo Ballester Valor MacLucasFFTW.c code.
 Improvement From Prime95.

 It also contains mfaktc code by Oliver Weihe and Eric Christenson
 adapted for CUDALucas use. Such code is under the GPL, and is noted as such.
 */

/* Include Files */
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "src/mpir/gmp.h"
#include <math.h>
#include <assert.h>
#include <time.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <float.h>

#ifdef _MSC_VER
//#define stat _stat
#define strncasecmp strnicmp // _strnicmp
#include <direct.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "parse.h"
#include "CUDAPm1.h"
#include "cuda/cuda_safecalls.h"
#include "rho.h"
#include "cuda/cuda_functions.h"
#include "lucas.h"
#include "bench.h"
#include "selftest.h"

double *g_ttmp, *g_ttp1;
double *g_x, *g_ct;
double *e_data;
double *rp_data;
int *g_xint;

char *size;
int threads1, threads2 = 128, threads3 = 128, device_number;
float *g_err, g_max_err = 0.0f;
int *g_datai, *g_carryi;
long long int *g_datal, *g_carryl;
cufftHandle plan;
cudaDeviceProp dev;

int fft_count;
int multipliers[250];
int quitting, checkpoint_iter, fftlen, tfdepth = 74, llsaved = 2, s_f, t_f, r_f, d_f, k_f;
int unused_mem = 100;
int polite, polite_f;
int b1 = 0, g_b1_commandline = 0;
int g_b2 = 0, g_b2_commandline = 0;
int g_d = 0, g_d_commandline = 0;
int g_e = 0;
int g_nrp = 0;
int g_eb1 = 0;
int keep_s1 = 0;
int selftest = 0;

char folder[132];
char input_filename[132], RESULTSFILE[132];
char INIFILE[132] = "CUDAPm1.ini";
char AID[132];  // Assignment key
char s_residue[32];

uint32 tiny_soe(uint32 limit, uint32 *primes) {
//simple sieve of erathosthenes for small limits - not efficient
//for large limits.
	uint8 *flags;
	uint16 prime;
	uint32 i, j;
	int it;
	
//allocate flags
	flags = (uint8 *) malloc(limit / 2 * sizeof(uint8));
	if (flags == NULL)
		printf("error allocating flags\n");
	memset(flags, 1, limit / 2);
	
//find the sieving primes, don't bother with offsets, we'll need to find those
//separately for each line in the main sieve.
	primes[0] = 2;
	it = 1;
	
//sieve using primes less than the sqrt of the desired limit
//flags are created only for odd numbers (mod2)
	for (i = 1; i < (uint32) (sqrt((double) limit) / 2 + 1); i++) {
		if (flags[i] > 0) {
			prime = (uint32) (2 * i + 1);
			for (j = i + prime; j < limit / 2; j += prime)
				flags[j] = 0;
			
			primes[it] = prime;
			it++;
		}
	}
	
//now find the rest of the prime flags and compute the sieving primes
	for (; i < limit / 2; i++) {
		if (flags[i] == 1) {
			primes[it] = (uint32) (2 * i + 1);
			it++;
		}
	}
	
	free(flags);
	return it;
}

int gtpr(int n, uint8* bprimes) {
	uint32 Nsmall = (uint32) sqrt((double) n);
	int numblocks;
	int primes_per_thread;
	uint32* primes;
	uint32* device_primes;
	uint32 np;
	uint8* results;
	
// find seed primes
	primes = (uint32*) malloc(Nsmall * sizeof(uint32));
	np = tiny_soe(Nsmall, primes);
	
// put the primes on the device
	cudaMalloc((void**) &device_primes, sizeof(uint32) * np);
	cudaMemcpy(device_primes, primes, sizeof(uint32) * np, cudaMemcpyHostToDevice);
	
// compute how many whole blocks we have to sieve and how many primes each
// thread will be responsible for.
	numblocks = (n / 3 / block_size + 1);
	primes_per_thread = ((np - startprime) + threadsPerBlock - 1) / threadsPerBlock;
	dim3 grid((uint32) sqrt((double) numblocks) + 1, (uint32) sqrt((double) numblocks) + 1);
	
	cudaMalloc((void**) &results, sizeof(uint8) * (n >> 1));
	cudaMemset(results, 0, sizeof(uint8) * (n >> 1));
	
	cudaAcc_SegSieve(grid, threadsPerBlock, device_primes, np, primes_per_thread, n, results);
	
	cutilSafeCall(cutilDeviceSynchronize());
	cutilSafeCall(cudaMemcpy(bprimes, results, sizeof(uint8) * (n >> 1), cudaMemcpyDeviceToHost));
	cudaFree(device_primes);
	cudaFree(results);
	free(primes);
	
	return 0;
}

/**************************************************************
 *
 *      FFT and other related Functions
 *
 **************************************************************/

void reset_err(float* maxerr, float value) {
	*maxerr *= value;
	cutilSafeCall(cudaMemcpy(g_err, maxerr, sizeof(float), cudaMemcpyHostToDevice));
}

void E_init_d(double *g, double value, int n) {
	double x[1] = {value};
	
	cutilSafeCall(cudaMemset(g, 0.0, sizeof(double) * n));
	cudaMemcpy(g, x, sizeof(double), cudaMemcpyHostToDevice);
}

void E_pre_mul(double *g_out, double *g_in, int n, int fft_f) {
	if (fft_f)
		cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_in, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
	cudaAcc_pre_mul(threads2, n, g_out, g_ct);
}

void E_mul(double *g_out, double *g_in1, double *g_in2, int n, float err, int fft_f) {
	
	if (fft_f)
		cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_in1, (cufftDoubleComplex *) g_in1, CUFFT_INVERSE));
	
	cudaAcc_mult3(threads2, n, g_out, g_in1, g_in2, g_ct);
	cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
	cudaAcc_norm1a(0, 0, threads1, n, g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, err);
	cudaAcc_norm2a(0, threads1, threads3, g_out, g_xint, n, g_datai, g_carryi, g_ttp1);
}

void E_sub_mul(double *g_out, double *g_in1, double *g_in2, double *g_in3, int n, float err, int chkpt) {
	
	cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_in1, (cufftDoubleComplex *) g_in1, CUFFT_INVERSE));
	cudaAcc_sub_mul(threads2, n, g_out, g_in1, g_in2, g_in3, g_ct);
	cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
	if (chkpt) {
		cudaAcc_norm1a(1, 0, threads1, n, g_out, g_datai, &g_xint[n], g_ttmp, g_carryi, g_err, err);
		cudaAcc_norm2a(1, threads1, threads3, g_out, &g_xint[n], n, g_datai, g_carryi, g_ttp1);
	} else {
		cudaAcc_norm1a(0, 0, threads1, n, g_out, g_datai, &g_xint[n], g_ttmp, g_carryi, g_err, err);
		cudaAcc_norm2a(0, threads1, threads3, g_out, &g_xint[n], n, g_datai, g_carryi, g_ttp1);
	}
}

void E_half_mul(double *g_out, double *g_in1, double *g_in2, int n, float err) {
	cudaAcc_mult2(threads2, n, g_out, g_in1, g_in2, g_ct);
	cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
	cudaAcc_norm1a(0, 0, threads1, n, g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, err);
	cudaAcc_norm2a(0, threads1, threads3, g_out, g_xint, n, g_datai, g_carryi, g_ttp1);
}

int E_to_the_p(double *g_out, double *g_in, mpz_t p, int n, int trans, float *err) {
// Assume g_in is premultiplied
	
	int last, j;
	int checksync = trans / (2 * 50) * 2 * 50;
	int checkerror = trans / (200) * 200;
	int checksave = trans / (2 * checkpoint_iter) * 2 * checkpoint_iter;
	int sync = 1;
	
	last = mpz_sizeinbase(p, 2);
	if (last == 1) {
		E_init_d(g_out, 1.0, n);
		if (mpz_tstbit(p, last - 1)) {
			cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
			cudaAcc_mult2(threads2, n, g_out, g_out, g_in, g_ct);
			cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
			cudaAcc_norm1a(0, 0, threads1, n, g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, *err);
			cudaAcc_norm2a(0, threads1, threads3, g_out, g_xint, n, g_datai, g_carryi, g_ttp1);
			trans += 2;
		}
		return trans;
	}
	
	cudaAcc_square1(threads2, n, g_out, g_in, g_ct);
	cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
	cudaAcc_norm1a(0, 0, threads1, n, g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, *err);
	cudaAcc_norm2a(0, threads1, threads3, g_out, g_xint, n, g_datai, g_carryi, g_ttp1);
	trans += 2;
	cutilSafeCall(cudaMemcpy(err, g_err, sizeof(float), cudaMemcpyDeviceToHost));
	if (mpz_tstbit(p, last - 2)) {
		cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
		cudaAcc_mult2(threads2, n, g_out, g_out, g_in, g_ct);
		cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
		cudaAcc_norm1a(0, 0, threads1, n, g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, *err);
		cudaAcc_norm2a(0, threads1, threads3, g_out, g_xint, n, g_datai, g_carryi, g_ttp1);
		trans += 2;
	}
	
	for (j = 3; j <= last && !quitting; j++) {
		cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
		cudaAcc_square(threads2, n, g_out, g_ct);
		cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
		cudaAcc_norm1a(0, 0, threads1, n, g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, *err);
		cudaAcc_norm2a(0, threads1, threads3, g_out, g_xint, n, g_datai, g_carryi, g_ttp1);
		trans += 2;
		if (mpz_tstbit(p, last - j)) {
			cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
			cudaAcc_mult2(threads2, n, g_out, g_out, g_in, g_ct);
			cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
			cudaAcc_norm1a(0, 0, threads1, n, g_out, g_datai, g_xint, g_ttmp, g_carryi, g_err, *err);
			cudaAcc_norm2a(0, threads1, threads3, g_out, g_xint, n, g_datai, g_carryi, g_ttp1);
			trans += 2;
		}
		if (trans - checkerror > 200) {
			sync = 0;
			checkerror += 200;
			cutilSafeCall(cudaMemcpy(err, g_err, sizeof(float), cudaMemcpyDeviceToHost));
			if (*err > 0.4)
				quitting = 2;
		}
		if (trans - checksave > 2 * checkpoint_iter) {
			checksave += 2 * checkpoint_iter;
			reset_err(err, 0.85f);
		}
		if (sync && polite_f && trans - checksync > 2 * polite) {
			checksync += 2 * polite;
			cutilSafeThreadSync();
		}
		sync = 1;
		fflush (NULL);
	}
	return trans;
}


void free_host(int *x_int) {
	free((char *) size);
	free((char *) x_int);
}

void free_gpu(void) {
	cufftSafeCall(cufftDestroy(plan));
	cutilSafeCall(cudaFree((char *) g_x));
	cutilSafeCall(cudaFree((char *) g_ct));
	cutilSafeCall(cudaFree((char *) g_xint));
	cutilSafeCall(cudaFree((char *) g_err));
	cutilSafeCall(cudaFree((char *) g_ttp1));
	cutilSafeCall(cudaFree((char *) g_ttmp));
	cutilSafeCall(cudaFree((char *) g_datai));
	cutilSafeCall(cudaFree((char *) g_datal));
	cutilSafeCall(cudaFree((char *) g_carryl));
	cutilSafeCall(cudaFree((char *) g_carryi));
}

void close_lucas(int *x_int) {
	free_host(x_int);
	free_gpu();
}

/**************************************************************************
 *                                                                        *
 *       End LL/GPU Functions, Begin Utility/CPU Functions                *
 *                                                                        *
 **************************************************************************/

int init_ffts() {
	//#define COUNT 139
	FILE *fft;
	char buf[132];
	int next_fft, j = 0, i = 0;
	int first_found = 0;
#define COUNT 160
	int default_mult[COUNT] = {  //this batch from GTX570 timings
		2, 8, 10, 14, 16, 18, 20, 32, 36, 42, 48, 50, 56, 60, 64, 70, 80, 84, 96,
		112, 120, 126, 128, 144, 160, 162, 168, 180, 192, 224, 256, 288, 320, 324,
		336, 360, 384, 392, 400, 448, 512, 576, 640, 648, 672, 720, 768, 784, 800,
		864, 896, 900, 1024, 1152, 1176, 1280, 1296, 1344, 1440, 1568, 1600, 1728,
		1792, 2048, 2160, 2304, 2352, 2592, 2688, 2880, 3024, 3136, 3200, 3584, 3600,
		4096, 4320, 4608, 4704, 5120, 5184, 5600, 5760, 6048, 6144, 6272, 6400, 6480,
		7168, 7200, 7776, 8064, 8192, 8640, 9216, 9408, 10240, 10368, 10584, 10800,
		11200, 11520, 12096, 12288, 12544, 12960, 13824, 14336, 14400, 16384, 17496,
		18144, 19208, 19600, 20000, 20250, 21952, 23328, 23814, 24300, 24500, 25088,
		25600, 26244, 27000, 27216, 28000, 28672, 31104, 31250, 32000, 32400, 32768,
		33614, 34992, 36000, 36288, 38416, 39200, 39366, 40500, 41472, 42336, 43200,
		43904, 47628, 49000, 50000, 50176, 51200, 52488, 54432, 55296, 56000, 57344,
		60750, 62500, 64000, 64800, 65536};
	
	char fftfile[256];
	char devname[256];
	remove_spaces(devname, dev.name);

	sprintf(fftfile, "%s_fft.txt", devname);
	fft = fopen(fftfile, "r");
	if (!fft) {
		printf("No %s file found. Using default fft lengths.\n", fftfile);
		printf("For optimal fft selection, please run\n");
		printf("./CUDAPm1 -cufftbench 1 8192 r\n");
		printf("for some small r, 0 < r < 6 e.g.\n");
		fflush (NULL);
		for (j = 0; j < COUNT; j++)
			multipliers[j] = default_mult[j];
	} else {
		while (fgets(buf, 132, fft) != NULL) {
			int le = 0;
			
			sscanf(buf, "%d", &le);
			if (next_fft = atoi(buf)) {
				if (!first_found) {
					while (i < COUNT && default_mult[i] < next_fft) {
						multipliers[j] = default_mult[i];
						i++;
						j++;
					}
					multipliers[j] = next_fft;
					j++;
					first_found = 1;
				} else {
					multipliers[j] = next_fft;
					j++;
				}
			}
		}
		while (default_mult[i] < multipliers[j - 1] && i < COUNT)
			i++;
		while (i < COUNT) {
			multipliers[j] = default_mult[i];
			j++;
			i++;
		}
		fclose(fft);
	}
	return j;
}


int fft_from_str(const char* str)
/* This is really just strtoul with some extra magic to deal with K or M */
{
	char* endptr;
	const char* ptr = str;
	int len, mult = 0;
	while (*ptr) {
		if (*ptr == 'k' || *ptr == 'K') {
			mult = 1024;
			break;
		}
		if (*ptr == 'm' || *ptr == 'M') {
			mult = 1024 * 1024;
			break;
		}
		ptr++;
	}
	if (!mult) {  // No K or M, treat as before    (PS The Python else clause on loops I mention in parse.c would be useful here :) )
		mult = 1;
	}
	len = (int) strtoul(str, &endptr, 10) * mult;
	if (endptr != ptr) {  // The K or M must directly follow the num (or the num must extend to the end of the str)
		fprintf(stderr, "can't parse fft length \"%s\"\n\n", str);
		exit(2);
	}
	return len;
}

//From apsen
void print_time_from_seconds(FILE *output, int sec) {
	if (sec > 3600) {
		fprintf(output, "%d", sec / 3600);
		sec %= 3600;
		fprintf(output, ":%02d", sec / 60);
	} else
		fprintf(output, "%d", sec / 60);
	sec %= 60;
	fprintf(output, ":%02d", sec);
}

void init_device(int device_number) {
	int device_count = 0;
	
	cudaGetDeviceCount(&device_count);
	if (device_number >= device_count) {
		printf("device_number >=  device_count ... exiting\n");
		printf("(This is probably a driver problem)\n\n");
		exit(2);
	}
	cudaSetDevice(device_number);
	cudaSetDeviceFlags (cudaDeviceBlockingSync);
	cudaGetDeviceProperties(&dev, device_number);
	// From Iain
	if (dev.major == 1 && dev.minor < 3) {
		printf("A GPU with compute capability >= 1.3 is required for double precision arithmetic\n\n");
		exit(2);
	}
	if (d_f) {
		printf("------- DEVICE %d -------\n", device_number);
		printf("name                %s\n", dev.name);
		printf("Compatibility       %d.%d\n", dev.major, dev.minor);
		printf("clockRate (MHz)     %d\n", dev.clockRate / 1000);
		printf("memClockRate (MHz)  %d\n", dev.memoryClockRate / 1000);
#ifdef _MSC_VER
		printf ("totalGlobalMem      %Iu\n", dev.totalGlobalMem);
#else
		printf("totalGlobalMem      %zu\n", dev.totalGlobalMem);
#endif
#ifdef _MSC_VER
		printf ("totalConstMem       %Iu\n", dev.totalConstMem);
#else
		printf("totalConstMem       %zu\n", dev.totalConstMem);
#endif
		printf("l2CacheSize         %d\n", dev.l2CacheSize);
#ifdef _MSC_VER
		printf ("sharedMemPerBlock   %Iu\n", dev.sharedMemPerBlock);
#else
		printf("sharedMemPerBlock   %zu\n", dev.sharedMemPerBlock);
#endif
		printf("regsPerBlock        %d\n", dev.regsPerBlock);
		printf("warpSize            %d\n", dev.warpSize);
#ifdef _MSC_VER
		printf ("memPitch            %Iu\n", dev.memPitch);
#else
		printf("memPitch            %zu\n", dev.memPitch);
#endif
		printf("maxThreadsPerBlock  %d\n", dev.maxThreadsPerBlock);
		printf("maxThreadsPerMP     %d\n", dev.maxThreadsPerMultiProcessor);
		printf("multiProcessorCount %d\n", dev.multiProcessorCount);
		printf("maxThreadsDim[3]    %d,%d,%d\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
		printf("maxGridSize[3]      %d,%d,%d\n", dev.maxGridSize[0], dev.maxGridSize[1], dev.maxGridSize[2]);
#ifdef _MSC_VER
		printf ("textureAlignment    %Iu\n", dev.textureAlignment);
#else
		printf("textureAlignment    %zu\n", dev.textureAlignment);
#endif
		printf("deviceOverlap       %d\n\n", dev.deviceOverlap);
	}
}

void rm_checkpoint(int q, int ks1) {
	char chkpnt_cfn[32];
	char chkpnt_tfn[32];
	
	if (!ks1) {
		sprintf(chkpnt_cfn, "c%ds1", q);
		sprintf(chkpnt_tfn, "t%ds1", q);
		(void) unlink(chkpnt_cfn);
		(void) unlink(chkpnt_tfn);
	}
	sprintf(chkpnt_cfn, "c%ds2", q);
	sprintf(chkpnt_tfn, "t%ds2", q);
	(void) unlink(chkpnt_cfn);
	(void) unlink(chkpnt_tfn);
}

int standardize_digits_int(int *x_int, int q, int n, int offset, int num_digits) {
	int j, digit, stop, qn = q / n, carry = 0;
	int temp;
	int lo = 1 << qn;
	int hi = lo << 1;
	
	digit = floor(offset * (n / (double) q));
	j = (n + digit - 1) % n;
	while (x_int[j] == 0 && j != digit)
		j = (n + j - 1) % n;
	if (j == digit && x_int[digit] == 0)
		return (1);
	else if (x_int[j] < 0)
		carry = -1;
	{
		stop = (digit + num_digits) % n;
		j = digit;
		do {
			x_int[j] += carry;
			carry = 0;
			if (size[j])
				temp = hi;
			else
				temp = lo;
			if (x_int[j] < 0) {
				x_int[j] += temp;
				carry = -1;
			}
			j = (j + 1) % n;
		} while (j != stop);
	}
	return (0);
}

void balance_digits_int(int* x, int q, int n) {
	int half_low = (1 << (q / n - 1));
	int low = half_low << 1;
	int high = low << 1;
	int upper, adj, carry = 0;
	int j;
	
	for (j = 0; j < n; j++) {
		if (size[j]) {
			upper = low;
			adj = high;
		} else {
			upper = half_low;
			adj = low;
		}
		x[j] += carry;
		carry = 0;
		if (x[j] >= upper) {
			x[j] -= adj;
			carry = 1;
		}
	}
	x[0] += carry;  // Good enough for our purposes.
}

unsigned *
read_checkpoint_packed(int q) {
	//struct stat FileAttrib;
	FILE *fPtr;
	unsigned *x_packed;
	char chkpnt_cfn[32];
	char chkpnt_tfn[32];
	int end = (q + 31) / 32;
	
	x_packed = (unsigned *) malloc(sizeof(unsigned) * (end + 25));
	
	if (selftest == 0) { /* Don't use checkpoints for self-tests */
		sprintf(chkpnt_cfn, "c%ds1", q);
		sprintf(chkpnt_tfn, "t%ds1", q);
		fPtr = fopen(chkpnt_cfn, "rb");
		if (!fPtr) {
//#ifndef _MSC_VER
//if(stat(chkpnt_cfn, &FileAttrib) == 0) fprintf (stderr, "\nUnable to open the checkpoint file. Trying the backup file.\n");
//#endif
			} else if (fread(x_packed, 1, sizeof(unsigned) * (end + 25), fPtr) != (sizeof(unsigned) * (end + 25))) {
				fprintf(stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
				fclose(fPtr);
			} else if (x_packed[end] != (unsigned int) q) {
				fprintf(stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
				fclose(fPtr);
			} else {
				fclose(fPtr);
				return x_packed;
			}
		fPtr = fopen(chkpnt_tfn, "rb");
		if (!fPtr) {
//#ifndef _MSC_VER
//    if(stat(chkpnt_cfn, &FileAttrib) == 0) fprintf (stderr, "\nUnable to open the backup file. Restarting test.\n");
//#endif
			} else if (fread(x_packed, 1, sizeof(unsigned) * (end + 25), fPtr) != (sizeof(unsigned) * (end + 25))) {
				fprintf(stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
				fclose(fPtr);
			} else if (x_packed[end] != (unsigned int) q) {
				fprintf(stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
				fclose(fPtr);
			} else {
				fclose(fPtr);
				return x_packed;
			}
	}

	x_packed[end] = q;
	x_packed[end + 1] = 0;  // n
	x_packed[end + 2] = 1;  // iteration number
	x_packed[end + 3] = 0;  // stage
	x_packed[end + 4] = 0;  // accumulated time
	x_packed[end + 5] = 0;  // b1
	// 6-9 reserved for extending b1
	// 10-24 reserved for stage 2
	int i;
	for (i = 6; i < 25; i++)
		x_packed[end + i] = 0;
	
	return x_packed;
}

int read_st2_checkpoint(int q, unsigned *x_packed) {
	//struct stat FileAttrib;

	if (selftest != 0) /* Don't use checkpoints for self-tests */
			return 0;

	FILE *fPtr;
	char chkpnt_cfn[32];
	char chkpnt_tfn[32];
	int end = (q + 31) / 32;
	
	sprintf(chkpnt_cfn, "c%ds2", q);
	sprintf(chkpnt_tfn, "t%ds2", q);
	fPtr = fopen(chkpnt_cfn, "rb");
	if (!fPtr) {
		// if(stat(chkpnt_cfn, &FileAttrib) == 0) fprintf (stderr, "\nUnable to open the checkpoint file. Trying the backup file.\n");
	} else if (fread(x_packed, 1, sizeof(unsigned) * (end + 25), fPtr) != (sizeof(unsigned) * (end + 25))) {
		fprintf(stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
		fclose(fPtr);
	} else if (x_packed[end] != (unsigned int) q) {
		fprintf(stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
		fclose(fPtr);
	} else {
		fclose(fPtr);
		return 1;
	}
	fPtr = fopen(chkpnt_tfn, "rb");
	if (!fPtr) {
		//if(stat(chkpnt_cfn, &FileAttrib) == 0) fprintf (stderr, "\nUnable to open the backup file. Restarting test.\n");
	} else if (fread(x_packed, 1, sizeof(unsigned) * (end + 25), fPtr) != (sizeof(unsigned) * (end + 25))) {
		fprintf(stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
		fclose(fPtr);
	} else if (x_packed[end] != (unsigned int) q) {
		fprintf(stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
		;
		fclose(fPtr);
	} else {
		fclose(fPtr);
		return 1;
	}
	return 0;
}

void pack_bits_int(int *x_int, unsigned *packed_x, int q, int n) {
	unsigned long long temp1, temp2 = 0;
	int i, j = 0, k = 0;
	int qn = q / n;
	
	for (i = 0; i < n; i++) {
		temp1 = x_int[i];
		temp2 += (temp1 << k);
		k += qn + size[i];
		if (k >= 32) {
			packed_x[j] = (unsigned) temp2;
			temp2 >>= 32;
			k -= 32;
			j++;
		}
	}
	packed_x[j] = (unsigned) temp2;
}

void set_checkpoint_data(unsigned *x_packed, int q, int n, int j, int stage, int time) {
	int end = (q + 31) / 32;
	
	x_packed[end + 0] = q;
	x_packed[end + 1] = n;
	x_packed[end + 2] = j;
	x_packed[end + 3] = stage;
	x_packed[end + 4] = time;
}

void write_checkpoint_packed(unsigned *x_packed, int q) {
	if (selftest != 0) /* Don't use checkpoints for self-tests */
			return;

	int end = (q + 31) / 32;
	FILE *fPtr;
	char chkpnt_cfn[32];
	char chkpnt_tfn[32];
	
	sprintf(chkpnt_cfn, "c%ds1", q);
	sprintf(chkpnt_tfn, "t%ds1", q);
	(void) unlink(chkpnt_tfn);
	(void) rename(chkpnt_cfn, chkpnt_tfn);
	fPtr = fopen(chkpnt_cfn, "wb");
	if (!fPtr) {
		fprintf(stderr, "Couldn't write checkpoint.\n");
		return;
	}
	fwrite(x_packed, 1, sizeof(unsigned) * (end + 25), fPtr);
	fclose(fPtr);
	if (s_f > 0)			// save all checkpoint files
			{
		char chkpnt_sfn[64];
#ifndef _MSC_VER
		sprintf(chkpnt_sfn, "%s/s" "%d.%d.%s", folder, q, x_packed[end + 2], s_residue);
#else
		sprintf (chkpnt_sfn, "%s\\s" "%d.%d.%s.txt", folder, q, x_packed[end + 2], s_residue);
#endif
		fPtr = fopen(chkpnt_sfn, "wb");
		if (!fPtr)
			return;
		fwrite(x_packed, 1, sizeof(unsigned) * (end + 25), fPtr);
		fclose(fPtr);
	}
}

void write_st2_checkpoint(unsigned *x_packed, int q) {
	if (selftest != 0) /* Don't use checkpoints for self-tests */
		return;

	int end = (q + 31) / 32;
	FILE *fPtr;
	char chkpnt_cfn[32];
	char chkpnt_tfn[32];
	
	sprintf(chkpnt_cfn, "c%ds2", q);
	sprintf(chkpnt_tfn, "t%ds2", q);
	(void) unlink(chkpnt_tfn);
	(void) rename(chkpnt_cfn, chkpnt_tfn);
	fPtr = fopen(chkpnt_cfn, "wb");
	if (!fPtr) {
		fprintf(stderr, "Couldn't write checkpoint.\n");
		return;
	}
	fwrite(x_packed, 1, sizeof(unsigned) * (end + 25), fPtr);
	fclose(fPtr);
	if (s_f > 0)			// save all checkpoint files
			{
		char chkpnt_sfn[64];
#ifndef _MSC_VER
		sprintf(chkpnt_sfn, "%s/s" "%d.%d.%s", folder, q, x_packed[end + 2], s_residue);
#else
		sprintf (chkpnt_sfn, "%s\\s" "%d.%d.%s.txt", folder, q, x_packed[end + 2], s_residue);
#endif
		fPtr = fopen(chkpnt_sfn, "wb");
		if (!fPtr)
			return;
		fwrite(x_packed, 1, sizeof(unsigned) * (end + 25), fPtr);
		fclose(fPtr);
	}
}

int printbits_int(int *x_int, int q, int n, int offset, FILE* fp, char *expectedResidue, int o_f) {
	int j, k = 0;
	int digit, bit;
	unsigned long long temp, residue = 0;
	
	digit = floor(offset * (n / (double) q));
	bit = offset - ceil(digit * (q / (double) n));
	j = digit;
	while (k < 64) {
		temp = x_int[j];
		residue = residue + (temp << k);
		k += q / n + size[j % n];
		if (j == digit) {
			k -= bit;
			residue >>= bit;
		}
		j = (j + 1) % n;
	}
	sprintf(s_residue, "%016llx", residue);
	
	printf("M%d, 0x%s,", q, s_residue);
	//if(o_f) printf(" offset = %d,", offset);
	printf(" n = %dK, %s", n / 1024, program);
	if (fp) {
		fprintf(fp, "M%d, 0x%s,", q, s_residue);
		if (o_f)
			fprintf(fp, " offset = %d,", offset);
		fprintf(fp, " n = %dK, %s", n / 1024, program);
	}
	return 0;
}

void unpack_bits_int(int *x_int, unsigned *packed_x, int q, int n) {
	unsigned long long temp1, temp2 = 0;
	int i, j = 0, k = 0;
	int qn = q / n;
	int mask1 = -1 << (qn + 1);
	int mask2;
	int mask;
	
	mask1 = ~mask1;
	mask2 = mask1 >> 1;
	for (i = 0; i < n; i++) {
		if (k < qn + size[i]) {
			temp1 = packed_x[j];
			temp2 += (temp1 << k);
			k += 32;
			j++;
		}
		if (size[i])
			mask = mask1;
		else
			mask = mask2;
		x_int[i] = ((int) temp2) & mask;
		temp2 >>= (qn + size[i]);
		k -= (qn + size[i]);
	}
}

void SetQuitting(int sig) {
	quitting = 1;
	sig == SIGINT ? printf("\tSIGINT") : (sig == SIGTERM ? printf("\tSIGTERM") : printf("\tUnknown signal"));
	printf(" caught, writing checkpoint.\n");
}

#ifndef _MSC_VER
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
int _kbhit(void) {
	struct termios oldt, newt;
	int ch;
	int oldf;
	
	tcgetattr(STDIN_FILENO, &oldt);
	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
	
	ch = getchar();
	
	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);
	
	if (ch != EOF) {
		ungetc(ch, stdin);
		return 1;
	}
	
	return 0;
}
#else
#include <conio.h>
#endif

int get_bit(int location, unsigned *control) {
	int digit = location / 32;
	int bit = location % 32;
	
	bit = 1 << bit;
	bit = control[digit] & bit;
	if (bit)
		bit /= bit;
	return (bit);
}

int round_off_test(int q, int n, int *j, unsigned *control, int last) {
	int k;
	float totalerr = 0.0;
	float terr, avgerr, maxerr = 0.0;
	float max_err = 0.0, max_err1 = 0.0;
	int bit;
	
	printf("Running careful round off test for 1000 iterations. If average error > 0.25, the test will restart with a longer FFT.\n");
	for (k = 0; k < 1000 && k < last; k++) {
		bit = get_bit(last - k - 1, control);
		terr = lucas_square(q, n, k, last, &maxerr, 1, bit, 1, k == 999);
		if (terr > maxerr)
			maxerr = terr;
		if (terr > max_err)
			max_err = terr;
		if (terr > max_err1)
			max_err1 = terr;
		totalerr += terr;
		reset_err(&maxerr, 0.85);
		if (terr >= 0.35) {
			printf("Iteration = %d < 1000 && err = %5.5f >= 0.35, increasing n from %dK\n", k, terr, n / 1024);
			fftlen++;
			return 1;
		}
		if (k && (k % 100 == 0)) {
			printf("Iteration  %d, average error = %5.5f, max error = %5.5f\n", k, totalerr / k, max_err);
			max_err = 0.0;
		}
	}
	avgerr = totalerr / 1000.0;
	if (avgerr > 0.25) {
		printf("Iteration 1000, average error = %5.5f > 0.25 (max error = %5.5f), increasing FFT length and restarting\n", avgerr, max_err);
		fftlen++;
		return 1;
	} else if (avgerr < 0) {
		fprintf(stderr, "Something's gone terribly wrong! Avgerr = %5.5f < 0 !\n", avgerr);
		exit(2);
	} else {
		printf("Iteration 1000, average error = %5.5f <= 0.25 (max error = %5.5f), continuing test.\n", avgerr, max_err1);
		reset_err(&maxerr, 0.85);
	}
	*j += 1000;
	return 0;
}

unsigned *get_control(int *j, int lim1, int lim2, int q) {
	mpz_t result;
	int p = 2;
	int limit;
	int prime_power = 1;
	unsigned *control = NULL;
	
	mpz_init(result);
	if (lim2 == 0) {
		mpz_set_ui(result, 2 * q);
		limit = lim1 / p;
		while (prime_power <= limit)
			prime_power *= p;
		mpz_mul_ui(result, result, prime_power);
		p = 3;
		while (p <= lim1) {
			while (p <= lim1 && !isprime(p))
				p += 2;
			limit = lim1 / p;
			prime_power = p;
			while (prime_power <= limit)
				prime_power *= p;
			mpz_mul_ui(result, result, prime_power);
			p += 2;
		}
	} else {
		p = lim1;
		if (!(lim1 & 1))
			p++;
		mpz_set_ui(result, 1);
		while (p <= lim2) {
			while (p <= lim2 && !isprime(p))
				p += 2;
			mpz_mul_ui(result, result, p);
			printf("prime_power: %d, %d\n", prime_power, p);
			p += 2;
		}
		
	}
	*j = mpz_sizeinbase(result, 2);
	control = (unsigned *) malloc(sizeof(unsigned) * ((*j + 31) / 32));
	mpz_export(control, NULL, -1, 4, 0, 0, result);
	mpz_clear(result);
	return control;
}

unsigned long long mpz2ull(mpz_t z) {
    unsigned long long result = 0;
    mpz_export(&result, 0, -1, sizeof result, 0, 0, z);
    return result;
}

int get_gcd(unsigned *x_packed, int q, int n, int stage) {
	mpz_t result, prime, prime1;
	int end = (q + 31) / 32;
	int rv = 0;
	
	mpz_init2(result, q);
	mpz_init2(prime, q);
	mpz_init2(prime1, q);
	mpz_import(result, end, -1, sizeof(x_packed[0]), 0, 0, x_packed);
	if (stage == 1)
		mpz_sub_ui(result, result, 1);
	mpz_setbit(prime, q);
	mpz_sub_ui(prime, prime, 1);
	if (mpz_cmp_ui(result, 0)) {
		mpz_gcd(prime1, prime, result);
		if (mpz_cmp_ui(prime1, 1)) {
			rv = 1;
			printf("M%d has a factor: ", q);
			mpz_out_str(stdout, 10, prime1);
			if (stage == 1)
				printf(" (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1, b1, g_e, n / 1024, program);  // Found in stage 1
			else
				printf(" (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1, g_b2, g_e, n / 1024, program);
			if (selftest == 0) {
				FILE* fp = fopen_and_lock(RESULTSFILE, "a");

				fprintf(fp, "M%d has a factor: ", q);
				mpz_out_str(fp, 10, prime1);
				if (AID[0] && strncasecmp(AID, "N/A", 3)) {
					if (stage == 1)
						fprintf(fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK, aid=%s %s)\n", b1, b1, g_e, n / 1024, AID, program);
					else
						fprintf(fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK, aid=%s %s)\n", b1, g_b2, g_e, n / 1024, AID, program);
				} else {
					if (stage == 1)
						fprintf(fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1, b1, g_e, n / 1024, program);
					else
						fprintf(fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1, g_b2, g_e, n / 1024, program);
				}
				unlock_and_fclose(fp);
			} else {
				st_result = mpz2ull(prime1);
			}
		}
	}
	if (rv == 0) {
		printf("M%d Stage %d found no factor", q, stage);
		printf(" (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1, g_b2, g_e, n / 1024, program);
		if (stage == 2 && selftest == 0) {
			FILE* fp = fopen_and_lock(RESULTSFILE, "a");
			fprintf(fp, "M%d found no factor", q);
			if (AID[0] && strncasecmp(AID, "N/A", 3))
				fprintf(fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK, aid=%s %s)\n", b1, g_b2, g_e, n / 1024, AID, program);
			else
				fprintf(fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1, g_b2, g_e, n / 1024, program);
			unlock_and_fclose(fp);
		}
		
	}
	mpz_clear(result);
	mpz_clear(prime);
	mpz_clear(prime1);
	return rv;
}

/* Analyze how well P-1 factoring will perform */

void guess_pminus1_bounds(int guess_exp, /* N in K*B^N+C. Exponent to test. */
int how_far_factored, /* Bit depth of trial factoring */
int tests_saved, /* 1 if doublecheck, 2 if first test */
int vals, int *bound1, int *bound2, double *success_rate) {
	int guess_B1, guess_B2, /*vals,*/i;
	double h, pass1_squarings, pass2_squarings;
	double logB1, logB2, kk, logkk, temp, logtemp, log2;
	double prob, ll_tests, numprimes;
	struct {
			int B1;
			int B2;
			double prob;
			double pass1_squarings;
			double pass2_squarings;
	} best[2];
	
	rho_init();
	
	for (i = 0; i < 2; i++) {
		best[i].B1 = 0;
		best[i].B2 = 0;
		best[i].prob = 0;
		best[i].pass1_squarings = 0;
		best[i].pass2_squarings = 0;
	}
	/* Guard against wild tests_saved values.  Huge values will cause this routine */
	/* to run for a very long time.  This shouldn't happen as auxiliaryWorkUnitInit */
	/* now has the exact same test. */

	if (tests_saved > 10)
		tests_saved = 10;
	
	/* Balance P-1 against 1 or 2 LL tests (actually more since we get a */
	/* corrupt result reported some of the time). */

	ll_tests = (double) tests_saved + 2 * 0.018;

	/* Compute how many temporaries we can use given our memory constraints. */
	/* Allow 1MB for code and data structures. */

// vals = cvt_mem_to_estimated_gwnums (max_mem (thread_num), k, b, n, c);
// if (vals < 1) vals = 1;
//vals = 176;
	/* Find the best B1 */

	log2 = log((double) 2.0);
	for (guess_B1 = 10000;; guess_B1 += 5000) {
		
		/* Constants */

		logB1 = log((double) guess_B1);
		
		/* Compute how many squarings will be required in pass 1 */

		pass1_squarings = ceil(1.44 * guess_B1);
		
		/* Try a lot of B2 values */

		for (guess_B2 = guess_B1; guess_B2 <= guess_B1 * 100; guess_B2 += guess_B1 >> 2) {
			
			/* Compute how many squarings will be required in pass 2.  In the
			   low-memory cases, assume choose_pminus1_plan will pick D = 210, E = 1
			   If more memory is available assume choose_pminus1_plan will pick
			   D = 2310, E = 2.  This will provide an accurate enough cost for our
			   purposes even if different D and E values are picked.  See
			   choose_pminus1_plan for a description of the costs of P-1 stage 2.

			   For cudapm1, we're not set up for e = 1, assume e = 2 in both cases */

			logB2 = log((double) guess_B2);
			numprimes = (unsigned long) (guess_B2 / (logB2 - 1.0) - guess_B1 / (logB1 - 1.0));
			if (guess_B2 <= guess_B1) {
				pass2_squarings = 0.0;
			} else if (vals <= 8) { /* D = 210, E = 1, passes = 48/temps */
				unsigned long num_passes;
				num_passes = (unsigned long) ceil(48.0 / (vals - 3));
				pass2_squarings = ceil((guess_B2 - guess_B1) / 210.0) * num_passes;
				pass2_squarings += numprimes * 1.1;
			} else {
				unsigned long num_passes;
				double numpairings;
				num_passes = (unsigned long) ceil(480.0 / (vals - 3));
				numpairings = (unsigned long) (numprimes / 2.0 * numprimes / ((guess_B2 - guess_B1) * 480.0 / 2310.0));
				pass2_squarings = 2400.0 + num_passes * 90.0; /* setup costs */
				pass2_squarings += ceil((guess_B2 - guess_B1) / 4620.0) * 2.0 * num_passes; /*number of base changes per pass * e with e = 2*/
				pass2_squarings += numprimes - numpairings; /*these are the sub_mul operations*/
			}
			
			/* Pass 2 FFT multiplications seem to be at least 20% slower than */
			/* the squarings in pass 1.  This is probably due to several factors. */
			/* These include: better L2 cache usage and no calls to the faster */
			/* gwsquare routine.  Nov, 2009:  On my Macbook Pro, with exponents */
			/* around 45M and using 800MB memory, pass2 squarings are 40% slower. */

			/* Owftheevil reports that CUDA squarings are only about 2% slower. */
			/* New normaliztion kernels benefit stage 1 more than stage 2, back to 9% */

			pass2_squarings *= 1.09;  // was 1.35
					
			/* What is the "average" value that must be smooth for P-1 to succeed? */
			/* Ordinarily this is 1.5 * 2^how_far_factored.  However, for Mersenne */
			/* numbers the factor must be of the form 2kp+1.  Consequently, the */
			/* value that must be smooth (k) is much smaller. */

			kk = 1.5 * pow(2.0, how_far_factored);
			// if (k == 1.0 && b == 2 && c == -1)
			kk = kk / 2.0 / guess_exp;
			logkk = log(kk);
			
			/* Set temp to the number that will need B1 smooth if k has an */
			/* average-sized factor found in stage 2 */

			temp = kk / ((guess_B1 + guess_B2) / 2);
			logtemp = log(temp);
			
			/* Loop over increasing bit lengths for the factor */

			prob = 0.0;
			for (h = how_far_factored;;) {
				double prob1, prob2;
				
				/* If temp < 1.0, then there are no factor to find in this bit level */

				if (logtemp > 0.0) {
					
					/* See how many smooth k's we should find using B1 */
					/* Using Dickman's function (see Knuth pg 382-383) we want k^a <= B1 */

					prob1 = rho(logB1 / logkk);
					
					/* See how many smooth k's we should find using B2 */
					/* Adjust this slightly to eliminate k's that have two primes > B1 and < B2 */
					/* Do this by assuming the largest factor is the average of B1 and B2 */
					/* and the remaining cofactor is B1 smooth */

					prob2 = prob1 + (rho(logB2 / logkk) - prob1) * (rho(logB1 / logtemp) / rho(logB2 / logtemp));
					if (prob2 < 0.0001)
						break;
					
					/* Add this data in to the total chance of finding a factor */

					prob += prob2 / (h + 1);
				}
				
				/* Move to next bit level */

				h += 1.0;
				logkk += log2;
				logtemp += log2;
			}
			
			/* See if this is a new best case scenario */

			if (guess_B2 == guess_B1
					|| prob * ll_tests * guess_exp - pass2_squarings > best[0].prob * ll_tests * guess_exp - best[0].pass2_squarings) {
				best[0].B2 = guess_B2;
				best[0].prob = prob;
				best[0].pass2_squarings = pass2_squarings;
				if (vals < 4)
					break;
				continue;
			}
			
			if (prob * ll_tests * guess_exp - pass2_squarings < 0.9 * (best[0].prob * ll_tests * guess_exp - best[0].pass2_squarings))
				break;
			continue;
		}
		
		/* Is this the best B1 thus far? */

		if (guess_B1 == 10000
				|| 1.15 * best[0].prob * ll_tests * guess_exp - (pass1_squarings + best[0].pass2_squarings)
						> 1.15 * best[1].prob * ll_tests * guess_exp - (best[1].pass1_squarings + best[1].pass2_squarings)) {
			best[1].B1 = guess_B1;
			best[1].B2 = best[0].B2;
			best[1].prob = best[0].prob;
			best[1].pass1_squarings = pass1_squarings;
			best[1].pass2_squarings = best[0].pass2_squarings;
			continue;
		}
		if (1.15 * best[0].prob * ll_tests * guess_exp - (pass1_squarings + best[0].pass2_squarings)
				< 0.9 * (1.15 * best[1].prob * ll_tests * guess_exp - (best[1].pass1_squarings + best[1].pass2_squarings)))
			break;
		continue;
	}
	
	/* Return the final best choice */

	if (1.15 * best[1].prob * ll_tests * guess_exp > best[1].pass1_squarings + best[1].pass2_squarings) {
		*bound1 = best[1].B1;
		*bound2 = best[1].B2;
// *squarings = (unsigned long)
//	(best[1].pass1_squarings +
//	 best[1].pass2_squarings);
		*success_rate = best[1].prob;
	} else {
		*bound1 = 10000;
		*bound2 = 250000;
// *squarings = 0;
		*success_rate = 0.0;
	}
}

/**************************************************************
 *
 *      Main Function
 *
 **************************************************************/
int stage2_init_param3(int e, int n, int trans, float *err) {
	int j, i, k = 0, base;
	mpz_t exponent;
	long long b[7];
	
	for (i = 0; i <= e / 2; i++) {
		base = 2 * i + 1;
		b[i] = 1;
		for (j = 0; j < e / 2; j++)
			b[i] *= base;
		b[i] *= b[i];
	}
	for (i = e / 2; i > 0; i--) {
		while (k < i) {
			j = i;
			while (j > k) {
				b[j] = b[j] - b[j - 1];
				j--;
			}
			k++;
			j = i;
			while (j > k) {
				b[j] = b[j] - b[j - 1];
				j--;
			}
		}
	}
	mpz_init(exponent);
	for (i = 0; i <= e / 2; i++) {
		mpz_set_ui(exponent, b[i]);
		trans = E_to_the_p(&e_data[2 * i * n], g_x, exponent, n, trans, err);
		if (i > 0) {
			cufftSafeCall(
					cufftExecZ2Z(plan, (cufftDoubleComplex *) &e_data[2 * i * n], (cufftDoubleComplex *) &e_data[2 * i * n],
							CUFFT_INVERSE));
			cudaAcc_copy(n, threads1, &e_data[(2 * i - 1) * n], &e_data[2 * i * n]);
			trans++;
		}
	}
	E_pre_mul(&e_data[e * n], &e_data[e * n], n, 0);
	E_pre_mul(&e_data[0], &e_data[0], n, 1);
	trans++;
	mpz_clear(exponent);
	return trans;
}

int next_base1(int k, int e, int n, int trans, float *err) {
	int j;
	
	if (k == 1)
		return (stage2_init_param3(e, n, trans, err));
	if (k > 3) {
		if (k <= e + 1) {
			E_mul(&e_data[(k - 3) * n], &e_data[(k - 2) * n], &e_data[(k - 3) * n], n, *err, 0);
			j = e + 3 - k;
			trans += 2 * (k - 3);
		} else {
			E_half_mul(&e_data[(e - 1) * n], &e_data[(e - 1) * n], &e_data[e * n], n, *err);
			j = 1;
			trans += 2 * (e - 1);
		}
		for (; j < e - 1; j++)
			E_mul(&e_data[(e - j - 1) * n], &e_data[(e - j) * n], &e_data[(e - j - 1) * n], n, *err, 1);
		cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) &e_data[1 * n], (cufftDoubleComplex *) &e_data[1 * n], CUFFT_INVERSE));
	}
	E_half_mul(&e_data[0], &e_data[1 * n], &e_data[0], n, *err);
	E_pre_mul(&e_data[0], &e_data[0], n, 1);
	trans += 2;
	return trans;
}

int stage2_init_param1(int k, int base, int e, int n, int trans, float *err) {
	int i, j;
	
	if (base > 1) {
		mpz_t exponent;
		mpz_init(exponent);
		mpz_ui_pow_ui(exponent, base, e);
		trans = E_to_the_p(&e_data[0], g_x, exponent, n, trans, err);
		E_pre_mul(g_x, &e_data[0], n, 1);
		trans++;
		mpz_clear(exponent);
	}
	
	if (k < 2 * e)
		for (j = 1; j <= k; j += 2) {
			trans = next_base1(j, e, n, trans, err);
			cutilSafeThreadSync();
		}
	else {
		mpz_t *exponents;
		
		exponents = (mpz_t *) malloc((e + 1) * sizeof(mpz_t));
		for (j = 0; j <= e; j++)
			mpz_init(exponents[j]);
		for (j = e; j >= 0; j--)
			mpz_ui_pow_ui(exponents[j], (k - j * 2), e);
		for (j = 0; j < e; j++)
			for (i = e; i > j; i--)
				mpz_sub(exponents[i], exponents[i - 1], exponents[i]);
		for (j = 0; j <= e; j++)
			trans = E_to_the_p(&e_data[j * n], g_x, exponents[j], n, trans, err);
		for (j = 0; j <= e; j++)
			mpz_clear(exponents[j]);
		E_pre_mul(&e_data[0], &e_data[0], n, 1);
		E_pre_mul(&e_data[e * n], &e_data[e * n], n, 1);
		for (j = 1; j < e; j++)
			cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *) &e_data[j * n], (cufftDoubleComplex *) &e_data[j * n], CUFFT_INVERSE));
		trans += e + 1;
		free(exponents);
	}
	return trans;
}

int stage2_init_param2(int num, int cur_rp, int base, int e, int n, uint8 *gaps, int trans, float *err) {
	int rp = 1, j = 0, i;
	mpz_t exponent;
	
	mpz_init(exponent);
	
	while (j < cur_rp) {
		j++;
		rp += 2 * gaps[j];
	}
	for (i = 0; i < num; i++) {
		mpz_ui_pow_ui(exponent, rp, e);
		trans = E_to_the_p(&rp_data[i * n], g_x, exponent, n, trans, err);
		E_pre_mul(&rp_data[i * n], &rp_data[i * n], n, 1);
		trans++;
		j++;
		if (rp < base - 1)
			rp += 2 * gaps[j];
	}
	
	mpz_clear(exponent);
	
	return trans;
}

int stage2_init_param4(int num, int cur_rp, int base, int e, int n, uint8 *gaps, int trans, float *err) {
	int rp = 1, j = 0, i, k = 1;
	
	while (j < cur_rp) {
		j++;
		rp += 2 * gaps[j];
	}
	trans = stage2_init_param1(rp, 1, e, n, trans, err);
	cudaAcc_copy(n, threads1, &rp_data[0], &e_data[0]);
	k = rp + 2;
	for (i = 1; i < num; i++) {
		j++;
		rp += 2 * gaps[j];
		while (k <= rp) {
			trans = next_base1(k, e, n, trans, err);
			cutilSafeThreadSync();
			k += 2;
		}
		cudaAcc_copy(n, threads1, &rp_data[i * n], &e_data[0]);
		
	}
	
	return trans;
}

int rp_init_count1(int k, int base, int e, int n) {
	int i, j, trans = 0;
	int numb[6] = {10, 38, 102, 196, 346, 534};
	int numb1[11] = {2, 8, 18, 32, 50, 72, 96, 120, 144, 168, 192};
	mpz_t exponent;
	
	mpz_init(exponent);
	mpz_ui_pow_ui(exponent, base, e);
	trans += (int) mpz_sizeinbase(exponent, 2) + (int) mpz_popcount(exponent) - 2;
	mpz_clear(exponent);
	
	if (k < 2 * e) {
		trans = 2 * trans + 1;
		trans += numb[e / 2 - 1] + numb1[k / 2 - 1];
		return (trans);
	} else {
		mpz_t *exponents;
		exponents = (mpz_t *) malloc((e + 1) * sizeof(mpz_t));
		for (j = 0; j <= e; j++)
			mpz_init(exponents[j]);
		for (j = e; j >= 0; j--)
			mpz_ui_pow_ui(exponents[j], (k - j * 2), e);
		for (j = 0; j < e; j++)
			for (i = e; i > j; i--)
				mpz_sub(exponents[i], exponents[i - 1], exponents[i]);
		for (j = 0; j <= e; j++) {
			trans += (int) mpz_sizeinbase(exponents[j], 2) + (int) mpz_popcount(exponents[j]) - 2;
		}
		for (j = 0; j <= e; j++)
			mpz_clear(exponents[j]);
		free(exponents);
		return 2 * (trans + e + 2) - 1;
	}
}

int rp_init_count1a(int k, int base, int e, int n) {
	int trans;
	int numb[6] = {10, 38, 102, 196, 346, 534};
	int numb1[12] = {0, 2, 8, 18, 32, 50, 72, 96, 120, 144, 168, 192};
	
	trans = (int) (e * log2((double) base) * 3.0);
	if (k < 2 * e) {
		trans += numb[e / 2 - 1] + numb1[(k + 1) / 2 - 1];
	} else {
		if (e == 2)
			trans += (int) (9.108 * log2((double) k) + 10.7);
		else if (e == 4)
			trans += (int) (30.349 * log2((double) k) + 50.5);
		else if (e == 6)
			trans += (int) (64.560 * log2((double) k) + 137.6);
		else if (e == 8)
			trans += (int) (110.224 * log2((double) k) + 265.2);
		else if (e == 10)
			trans += (int) (168.206 * log2((double) k) + 478.6);
		else
			trans += (int) (237.888 * log2((double) k) + 731.5);
	}
	return trans;
}

int rp_init_count2(int num, int cur_rp, int e, int n, uint8 *gaps) {
	int rp = 1, j = 0, i, trans = 0;
	int numb[6] = {10, 38, 102, 196, 346, 534};
	
	while (j < cur_rp) {
		j++;
		rp += 2 * gaps[j];
	}
	if (cur_rp == 0)
		trans -= e * e / 2 - 1;
	cur_rp = rp;
	if (rp == 1)
		trans += numb[e / 2 - 1];
	else
		trans = rp_init_count1(rp, 1, e, n);
	for (i = 1; i < num; i++) {
		j++;
		rp += 2 * gaps[j];
	}
	trans += e * (rp - cur_rp);
	
	return trans;
}

int rp_init_count2a(int cur_rp, int e, int n, uint8 *gaps) {
	int rp = 1, j = 0, trans = 0;
	int numb[6] = {10, 38, 102, 196, 346, 534};
	
	while (j < cur_rp) {
		j++;
		rp += 2 * gaps[j];
	}
	if (cur_rp == 0)
		trans -= e * e / 2 - 1;
	cur_rp = rp;
	if (rp == 1)
		trans += numb[e / 2 - 1];
	else
		trans = rp_init_count1a(rp, 1, e, n);
	
	return trans;
}

int stage2(int *x_int, unsigned *x_packed, int q, int n, int nrp, float err) {
	int j, i = 0, t;
	int e, d, b2 = g_b2;
	int rpt = 0, rp;
	int ks, ke, m = 0, k;
	int last = 0;
	uint8 *bprimes = NULL;
	int prime, prime_pair;
	uint8 *rp_gaps = NULL;
	int sprimes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 43, 47, 53, 0};
	uint8 two_to_i[] = {1, 2, 4, 8, 16, 32, 64, 128};
	int count0 = 0, count1 = 0, count2 = 0;
	mpz_t control;
	timeval time0, time1;
	
	{
		int best_guess = 0x01111111;
		int best_d = 0, best_e = 0, best_nrp = 0;
		int guess;
		int passes;
		int su;
		int nrpe = 0;
		int start_e = 2, end_e = 12;
		int start_d = 9240, d_div = 1;
		
		if (g_e) {
			start_e = g_e;
			end_e = g_e;
		}
		if (g_d) {
			start_d = g_d;
			d_div = g_d;
		}
		for (d = start_d; d > 1; d /= d_div) {
			if (d >= 2310) {
				rpt = d / 2310 * 480;
				i = 4;
			} else if (d >= 210) {
				rpt = d / 210 * 48;
				i = 3;
			} else if (d >= 30) {
				rpt = d / 30 * 8;
				i = 2;
			}
//else if(d >= 6)
// {
//  rpt = d / 6 * 2;
//  i = 1;
//}
			if (b1 * sprimes[i] * 53 < b2)
				ks = ((((b1 * 53 + 1) >> 1) + d - 1) / d - 1) * d;
			else if (b1 * sprimes[i] < b2)
				ks = ((((b2 / sprimes[i] + 1) >> 1) + d - 1) / d - 1) * d;
			else
				ks = ((((b1 + 1) >> 1) + d - 1) / d - 1) * d;
			ke = ((((b2 + 1) >> 1) + d - 1) / d) * d;
			ks = ((ks / d) << 1) + 1;
			ke = (ke / d) << 1;
			
			for (e = start_e; e <= end_e; e += 2) {
				nrpe = nrp - e - 1;
				if (nrpe <= 0)
					break;
				passes = (rpt + nrpe - 1) / nrpe;
				while (nrpe > 1 && passes == (rpt + nrpe - 2) / (nrpe - 1))
					nrpe--;
				guess = rp_init_count1a(ks, d, e, n) * passes;
				for (su = 0; su < rpt; su += nrpe)
					guess += rp_init_count1a((su * d / rpt) | 1, 1, e, n);
				guess += 2 * e * (d / 2 - passes) - e * e / 2;
				double numprimes = (double) ke * d / (log((double) ke * d) - 1.0) - (double) b1 / (log((double) b1) - 1.0);
				double numpairings = numprimes / 2.0 * numprimes / ((double) ((ke - ks) * d) * (double) rpt / d);
				guess += e * (ke - ks) * passes + (2.2) * (int) (numprimes - numpairings);
				if (e == 4)
					guess = (int) guess * 0.87;
				if (e == 6)
					guess = (int) guess * 0.82;
				if (e == 12)
					guess = (int) guess * 0.75;
				if (guess < best_guess) {
					best_guess = guess;
					best_d = d;
					best_e = e;
					best_nrp = nrpe;
				}
			}
			if (d > 2310)
				d -= 2310;
			else if (d > 210)
				d -= 210;
			else if (d >= 30)
				d -= 30;
//else if(d>=6) d -= 6;
		}
		d = best_d;
		e = best_e;
		nrp = best_nrp;
	}
	if (d == 0)
		exit(3);
	
	int end = (q + 31) / 32;
	if (x_packed[end + 10] == 0) {
		x_packed[end + 10] = b2;
		x_packed[end + 11] = d;
		x_packed[end + 12] = e;
		x_packed[end + 13] = nrp;
		x_packed[end + 14] = 0;  // m = number of relative primes already finished
		x_packed[end + 15] = 0;  // k = how far done with currect crop of relative primes
		x_packed[end + 16] = 0;  // t = where to find next relativel prime in the bit array
		x_packed[end + 17] = 0;  // extra initialization transforms from starting in the middle of a pass
	} else {
		b1 = x_packed[end + 5];
		b2 = x_packed[end + 10];
		d = x_packed[end + 11];
		e = x_packed[end + 12];
		nrp = x_packed[end + 13];
	}
	g_e = e;
	printf("Using b1 = %d, b2 = %d, d = %d, e = %d, nrp = %d\n", b1, b2, d, e, nrp);
	
	if (d % 2310 == 0) {
		i = 4;
		rpt = 480 * d / 2310;
	} else if (d % 210 == 0) {
		i = 3;
		rpt = 48 * d / 210;
	} else if (d % 30 == 0) {
		i = 2;
		rpt = 8 * d / 30;
	} else {
		i = 1;
		rpt = 2 * d / 6;
	}
	
	if (b1 * sprimes[i] * 53 < b2)
		ks = ((((b1 * 53 + 1) >> 1) + d - 1) / d - 1) * d;
	else if (b1 * sprimes[i] < b2)
		ks = ((((b2 / sprimes[i] + 1) >> 1) + d - 1) / d - 1) * d;
	else
		ks = ((((b1 + 1) >> 1) + d - 1) / d - 1) * d;
	ke = ((((b2 + 1) >> 1) + d - 1) / d) * d;
	
	bprimes = (uint8*) malloc(ke * sizeof(uint8));
	if (!bprimes) {
		printf("failed to allocate bprimes\n");
		exit(1);
	}
	for (j = 0; j < ke; j++)
		bprimes[j] = 0;
	gtpr(2 * ke, bprimes);
	
	// Make sure previous function is finished before using the results...
	cutilSafeCall(cudaDeviceSynchronize());

	for (j = 0; j < 10; j++)
		bprimes[j] = 1;
	bprimes[0] = bprimes[4] = bprimes[7] = 0;

#ifdef EBUG
	size_t global_mem, free_mem;
	cudaMemGetInfo(&free_mem, &global_mem);
	printf("Total: %zuM\tFree: %zuM\n", global_mem / 1024 / 1024, free_mem / 1024 / 1024);

	printf("rp_data: Attempting to cudaMalloc %d MB\n", sizeof(double) * n * nrp / 1024 / 1024);
#endif
	cutilSafeCall(cudaMalloc((void **) &rp_data, sizeof(double) * n * nrp));
#ifdef EBUG
	printf("e_data: Attempting to cudaMalloc %d MB\n", sizeof(double) * n * (e + 1) /1024 / 1024);
#endif
	cutilSafeCall(cudaMalloc((void **) &e_data, sizeof(double) * n * (e + 1)));
	
	for (j = (b1 + 1) >> 1; j < ks; j++) {
		if (bprimes[j] == 1) {
			m = i;
			last = j;
			while (sprimes[m]) {
				prime = sprimes[m] * j + (sprimes[m] >> 1);
				m++;
				if (prime < ks)
					continue;
				if (prime > ke)
					break;
				prime_pair = prime + d - 1 - ((prime % d) << 1);
				bprimes[last] = 0;
				bprimes[prime] = 1;
				if (bprimes[prime_pair])
					break;
				last = prime;
			}
		}
	}
	
	rp_gaps = (uint8*) malloc(rpt * sizeof(uint8));
	if (!rp_gaps) {
		printf("failed to allocate rp_gaps\n");
		exit(1);
	}
	j = 0;
	k = 0;
	
	for (rp = 1; rp < d; rp += 2) {
		k++;
		for (m = 0; m < i; m++)
			if ((rp % sprimes[m]) == 0)
				break;
		if (m == i) {
			rp_gaps[j] = k;
			j++;
			k = 0;
		}
	}
	
	k = ks + (d >> 1);
	m = k - 1;
	j = 0;
	rp = 0;
	uint8 *tprimes = (uint8*) malloc(rpt / 8 * sizeof(uint8));
	int l = 0;
	while (m < ke) {
		tprimes[l] = 0;
		for (i = 0; i < 8; i++) {
			m += rp_gaps[j];
			k -= rp_gaps[j];
			if (bprimes[m] || bprimes[k]) {
				tprimes[l] |= two_to_i[i];
				count1++;
			} else
				count0++;
			if (bprimes[m] && bprimes[k])
				count2++;
			j++;
			if (j == rpt) {
				j = 0;
				m += (d >> 1);
				k = m + 1;
			}
		}
		l++;
		if (l * 8 == rpt) {
			for (t = 0; t < (rpt >> 3); t++)
				bprimes[rp + t] = tprimes[t];
			l = 0;
			rp += rpt >> 3;
		}
	}
	free(tprimes);
	printf("Zeros: %d, Ones: %d, Pairs: %d\n", count0, count1, count2);
	
	mpz_init(control);
	mpz_import(control, (ke - ks) / d * rpt / sizeof(bprimes[0]), -1, sizeof(bprimes[0]), 0, 0, bprimes);
	free(bprimes);
	
	unpack_bits_int(x_int, x_packed, q, n);
	balance_digits_int(x_int, q, n);
	cudaMemcpy(&g_xint[n], x_int, sizeof(int) * n, cudaMemcpyHostToDevice);
	
	int fp = 1;
	int num_tran = 0, temp_tran;
	int tran_save;
	int itran_tot;
	int ptran_tot;
	int itran_done = 0;
	int ptran_done = 0;
	double checkpoint_int, checkpoint_bnd;
	double time, ptime = 0.0, itime = 0.0;
	
	ks = ((ks / d) << 1) + 1;
	ke = (ke / d) << 1;
	m = x_packed[end + 14];
	k = x_packed[end + 15];
	t = x_packed[end + 16];
	if (m + k > 0)  // some stage 2 has already been done
			{
		itran_done = x_packed[end + 18] + x_packed[end + 17];
		ptran_done = x_packed[end + 19];
		itime = x_packed[end + 20];
		ptime = x_packed[end + 21];
	}
	
	ptran_tot = (ke - ks - 1) * e * ((rpt + nrp - 1) / nrp) + count1 * 2;
	int passes;
	passes = (rpt + nrp - 1) / nrp;
	itran_tot = rp_init_count1(ks, d, e, n) * passes + x_packed[end + 17];
	int su = 0;
	while (su < rpt) {
		if (rpt - su > nrp) {
			itran_tot += rp_init_count2(nrp, su, e, n, rp_gaps);
		} else {
			itran_tot += rp_init_count2(rpt - su, su, e, n, rp_gaps);
		}
		su += nrp;
	}
	
	if (k == 0)
		k = ks;
	if (nrp > rpt - m)
		nrp = rpt - m;
	gettimeofday(&time0, NULL);
	do {
		printf("Processing %d - %d of %d relative primes.\n", m + 1, m + nrp, rpt);
		printf("Initializing pass... ");
		cudaAcc_apply_weights(n, threads1, g_x, &g_xint[0], g_ttmp);
		E_pre_mul(g_x, g_x, n, 1);
		num_tran = stage2_init_param4(nrp, m, d, e, n, rp_gaps, num_tran, &err);
		temp_tran = num_tran;
		num_tran = stage2_init_param1(k, d, e, n, num_tran, &err);
		cudaAcc_apply_weights(n, threads1, g_x, &g_xint[n], g_ttmp);
		temp_tran = num_tran - temp_tran;
		itran_done += num_tran;
		if ((m > 0 || k > ks) && fp) {
			x_packed[end + 17] += num_tran;
			itran_tot += num_tran;
		}
		fp = 0;
		cutilSafeCall(cudaMemcpy(&err, g_err, sizeof(float), cudaMemcpyDeviceToHost));
		gettimeofday(&time1, NULL);
		time = 1000000.0 * (double) (time1.tv_sec - time0.tv_sec) + time1.tv_usec - time0.tv_usec;
		itime += time / 1000000.0;
		
		if (!quitting) {
			printf("done. transforms: %d, err = %0.5f, (%0.2f real, %0.4f ms/tran,  ETA ", num_tran, err, time / 1000000.0,
					time / (float) (num_tran * 1000));
			if (m == 0 && k == ks)
				printf("NA");
			else
				print_time_from_seconds(stdout,
						(int) (itime * ((double) itran_tot / itran_done - 1) + ptime * ((double) ptran_tot / ptran_done - 1)));
		}
		printf(")\n");
		
		time0.tv_sec = time1.tv_sec;
		time0.tv_usec = time1.tv_usec;
		num_tran = 0;
		tran_save = 0;
		
		checkpoint_int = (ke - ks) / 2 * e + nrp * ((long int) count1 / rpt);
		int chkp_per_pass;
		chkp_per_pass = RINT_x86(checkpoint_int / checkpoint_iter);
		if (chkp_per_pass == 0)
			chkp_per_pass = 1;
		int next_checkpoint = ke - 1;
		checkpoint_int = (ke - ks + 1) / (double) chkp_per_pass;
		checkpoint_bnd = ks - 2.0;
		while ((int) checkpoint_bnd < k)
			checkpoint_bnd += checkpoint_int;
		next_checkpoint = RINT_x86(checkpoint_bnd);
		next_checkpoint |= 1;
		
		for (; k < ke && !quitting; k += 2) {
			int t_last = -1;
			{
				i = nrp - 1;
				while (i && !mpz_tstbit(control, t + i))
					i--;
				if (i)
					t_last = t + i;
			}
			for (j = 0; j < nrp; j++) {
				if (mpz_tstbit(control, t)) {
					E_sub_mul(g_x, g_x, &e_data[0], &rp_data[j * n], n, err, t == t_last);
					num_tran += 2;
					if (num_tran % 200 == 0 || t_f) {
						cutilSafeCall(cudaMemcpy(&err, g_err, sizeof(float), cudaMemcpyDeviceToHost));
						if (err > 0.4)
							quitting = 2;
					} else if (polite_f && num_tran % (2 * polite) == 0)
						cutilSafeThreadSync();
				}
				t++;
			}
			_kbhit();
			t += rpt - nrp;
			if (!quitting) {
				if (k < ke - 1)
					num_tran = next_base1(k, e, n, num_tran, &err);
				if (num_tran % 200 < 2 * e || t_f) {
					cutilSafeCall(cudaMemcpy(&err, g_err, sizeof(float), cudaMemcpyDeviceToHost));
					if (err > 0.4)
						quitting = 2;
				} else if (polite_f && num_tran % (2 * polite) < 2 * e)
					cutilSafeThreadSync();
			}
			if ((k == next_checkpoint || quitting == 1) && quitting != 2) {
				checkpoint_bnd += checkpoint_int;
				next_checkpoint = RINT_x86(checkpoint_bnd);
				next_checkpoint |= 1;
				if (quitting == 1)
					cutilSafeCall(cudaMemcpy(&err, g_err, sizeof(float), cudaMemcpyDeviceToHost));
				if (err <= 0.4f) {
					cutilSafeCall(cudaMemcpy(x_int, &g_xint[n], sizeof(int) * n, cudaMemcpyDeviceToHost));
					standardize_digits_int(x_int, q, n, 0, n);
					pack_bits_int(x_int, x_packed, q, n);
					x_packed[end + 13] = nrp;
					if (k < ke - 1) {
						x_packed[end + 14] = m;
						x_packed[end + 15] = k + 2;
						x_packed[end + 16] = t;
					} else {
						x_packed[end + 14] = m + nrp;
						x_packed[end + 15] = ks;
						x_packed[end + 16] = m + nrp;
					}
					gettimeofday(&time1, NULL);
					time = 1000000.0 * (double) (time1.tv_sec - time0.tv_sec) + time1.tv_usec - time0.tv_usec;
					ptime += time / 1000000.0;
					x_packed[end + 18] = itran_done;
					x_packed[end + 19] = ptran_done + num_tran;
					x_packed[end + 20] = itime;
					x_packed[end + 21] = ptime;
					write_st2_checkpoint(x_packed, q);
					printf("Transforms: %5d ", num_tran - tran_save);
					printbits_int(x_int, q, n, 0, 0, NULL, 0);
					printf(" err = %5.5f (", err);
					print_time_from_seconds(stdout, (int) time1.tv_sec - time0.tv_sec);
					printf(" real, %4.4f ms/tran, ETA ", time / 1000.0 / (num_tran - tran_save));
					print_time_from_seconds(stdout,
							(int) itime * ((double) itran_tot / itran_done - 1)
									+ ptime * ((double) ptran_tot / (ptran_done + num_tran) - 1));
					printf(")\n");
					fflush (stdout);
					tran_save = num_tran;
					time0.tv_sec = time1.tv_sec;
					time0.tv_usec = time1.tv_usec;
					reset_err(&err, 0.85f);
				}
			}
		}
		k = ks;
		m += nrp;
		t = m;
		if (rpt - m < nrp)
			nrp = rpt - m;
		ptran_done += num_tran;
		num_tran = 0;
		printf("\n");
	} while (m < rpt && !quitting);
	if (quitting < 2) {
		if (!quitting)
			printf("Stage 2 complete, %d transforms, estimated total time = ", ptran_done + itran_done);
		else
			printf("Quitting, estimated time spent = ");
		print_time_from_seconds(stdout, (int) itime + ptime);
		printf("\n");
	} else if (quitting == 2)
		printf("err = %5.5g >= 0.40, quitting.\n", err);
	free(rp_gaps);
	cutilSafeCall(cudaFree((char *) e_data));
	cutilSafeCall(cudaFree((char *) rp_data));
	mpz_clear(control);
	return 0;
}

int check_pm1(int q, char *expectedResidue) {
	int n, j, last = 0;
	int error_flag, checkpoint_flag;
	int *x_int = NULL;
	unsigned *x_packed = NULL;
	float maxerr = 0.0f, terr;
	int restarting = 0;
	timeval time0, time1;
	int total_time = 0, start_time;
	int j_resume = 0;
	int bit;
	unsigned *control = NULL;
	int stage = 0, st1_factor = 0;
	size_t global_mem, free_mem, use_mem;
	int nrp = g_nrp;
	
	signal(SIGTERM, SetQuitting);
	signal(SIGINT, SetQuitting);
	
	cudaMemGetInfo(&free_mem, &global_mem);
#ifdef _MSC_VER
	printf("CUDA reports %IuM of %IuM GPU memory free.\n",free_mem/1024/1024, global_mem/1024/1024);
#else
	printf("CUDA reports %zuM of %zuM GPU memory free.\n", free_mem / 1024 / 1024, global_mem / 1024 / 1024);
#endif
	
	do { /* while (restarting) */
		maxerr = 0.0;
		
		if (stage == 0) {
			if (!x_packed)
				x_packed = read_checkpoint_packed(q);
			x_int = init_lucas_packed_int(x_packed, q, &n, &j, &stage, &total_time);
		}
		if (!x_int)
			exit(2);
		if (stage == 2) {
			if (read_st2_checkpoint(q, x_packed)) {
				printf("Stage 2 checkpoint found.\n");
				int end = (q + 31) / 32;
				b1 = x_packed[end + 5];
			} else
				printf("No stage 2 checkpoint.\n");
		}
		
		g_d = g_d_commandline;
		if (g_nrp == 0)
			nrp = ((free_mem - (size_t) unused_mem * 1024 * 1024) / n / 8 - 7);
#ifdef _MSC_VER
		if (nrp > (4096/sizeof(double))*1024*1024/n)
			nrp = (4096/sizeof(double))*1024*1024/n;  // Max single allocation of 4 GB on Windows?
#endif
		if (nrp < 4)
			nrp = 4;
		use_mem = (size_t)(8 * (nrp + 7) * (size_t) n);
#ifdef _MSC_VER
		printf("Using up to %IuM GPU memory.\n",use_mem/1024/1024);
#else
		printf("Using up to %zuM GPU memory.\n", use_mem / 1024 / 1024);
#endif
		if (free_mem < use_mem)
			printf("WARNING:  There may not be enough GPU memory for stage 2!\n");
		
		double successrate = 0.0;
		if ((g_b1_commandline == 0) || (g_b2_commandline == 0)) {
			guess_pminus1_bounds(q, tfdepth, llsaved, nrp, &b1, &g_b2, &successrate);
		}
		if (g_b1_commandline > 0)
			b1 = g_b1_commandline;
		if (g_b2_commandline > 0)
			g_b2 = g_b2_commandline;
		if ((g_b1_commandline == 0) && (g_b2_commandline == 0))
			printf("Selected B1=%d, B2=%d, %0.3g%% chance of finding a factor\n", b1, g_b2, successrate * 100);
		
		if (x_packed[(q + 31) / 32 + 5] == 0 || restarting)
			x_packed[(q + 31) / 32 + 5] = b1;
		else {
			b1 = x_packed[(q + 31) / 32 + 5];
			printf("Using B1 = %d from savefile.\n", b1);
			fflush (stdout);
		}
		
		if (g_b2 > MAX_B2)
			printf("WARNING:  Expected failure with B2 > %d!\n", MAX_B2);  //max B2 supported?
		fflush (stdout);
		
		if (stage == 1) {
			if (control)
				free(control);
			control = get_control(&last, b1, 0, q);
		}
		gettimeofday(&time0, NULL);
		start_time = time0.tv_sec;
		
		restarting = 0;
		if (j == 1) {
			printf("Starting stage 1 P-1, M%d, B1 = %d, B2 = %d, fft length = %dK\n", q, b1, g_b2, n / 1024);
			printf("Doing %d iterations\n", last);
//restarting = round_off_test(q, n, &j, control, last);
//if(restarting) stage = 0;
		} else {
			if (stage == 1) {
				printf("Continuing stage 1 from a partial result of M%d fft length = %dK, iteration = %d\n", q, n / 1024, j);
				j_resume = j % checkpoint_iter - 1;
			} else {
				printf("Continuing stage 2 from a partial result of M%d fft length = %dK\n", q, n / 1024);
			}
		}
		fflush(stdout);
		
		for (; !restarting && j <= last; j++)  // Main LL loop
				{
			if ((j % 100) == 0 || t_f)
				error_flag = 1;
			else
				error_flag = 0;
			if ((j % checkpoint_iter == 0) || j == last)
				checkpoint_flag = 1;
			else
				checkpoint_flag = error_flag;
			bit = get_bit(last - j, control);
			terr = lucas_square(q, n, j, last, &maxerr, error_flag, bit, stage, checkpoint_flag);
			if (quitting == 1 && !checkpoint_flag) {
				j++;
				bit = get_bit(last - j, control);
				terr = lucas_square(q, n, j, last, &maxerr, 1, bit, stage, 1);
			}
			if (error_flag || quitting == 1) {
				if (terr >= 0.40) {
					printf("Iteration = %d, err = %5.5g >= 0.40, quitting.\n", j, terr);
					quitting = 2;
				}
			}
			if ((j % checkpoint_iter) == 0 || quitting) {
				if (quitting < 2) {
					cutilSafeCall(cudaMemcpy(x_int, g_xint, sizeof(int) * n, cudaMemcpyDeviceToHost));
					standardize_digits_int(x_int, q, n, 0, n);
					gettimeofday(&time1, NULL);
					total_time += (time1.tv_sec - start_time);
					start_time = time1.tv_sec;
					set_checkpoint_data(x_packed, q, n, j + 1, stage, total_time);
					pack_bits_int(x_int, x_packed, q, n);
					write_checkpoint_packed(x_packed, q);
				}
				if (quitting == 0) {
					printf("Iteration %d ", j);
					printbits_int(x_int, q, n, 0, 0, NULL, 0);
					long long diff = time1.tv_sec - time0.tv_sec;
					long long diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
					long long diff2 = (last - j) * diff1 / ((checkpoint_iter - j_resume) * 1e6);
					gettimeofday(&time0, NULL);
					printf(" err = %5.5f (", maxerr);
					print_time_from_seconds(stdout, (int) diff);
					printf(" real, %4.4f ms/iter, ETA ", diff1 / 1000.0 / (checkpoint_iter - j_resume));
					print_time_from_seconds(stdout, (int) diff2);
					printf(")\n");
					fflush(stdout);
					if (j_resume)
						j_resume = 0;
					reset_err(&maxerr, 0.85);  // Instead of tracking maxerr over whole run, reset it at each checkpoint.
				} else {
					printf("Estimated time spent so far: ");
					print_time_from_seconds(stdout, total_time);
					printf("\n\n");
					j = last + 1;
				}
			}
			if (k_f && !quitting && (!(j & 15)) && _kbhit())
				interact();  // abstracted to clean up check()
			fflush(stdout);
		}
		
		if (!restarting && !quitting) {  // done with stage 1
			if (stage == 1) {
				free((char *) control);
				gettimeofday(&time1, NULL);
				cutilSafeCall(cudaMemcpy(x_int, g_xint, sizeof(int) * n, cudaMemcpyDeviceToHost));
				standardize_digits_int(x_int, q, n, 0, n);
				if (g_eb1 > b1)
					stage = 3;
				else if (g_b2 > b1)
					stage = 2;
				set_checkpoint_data(x_packed, q, n, j + 1, stage, total_time);
				pack_bits_int(x_int, x_packed, q, n);
				write_checkpoint_packed(x_packed, q);
				printbits_int(x_int, q, n, 0, NULL, 0, 1);
				total_time += (time1.tv_sec - start_time);
				printf("\nStage 1 complete, estimated total time = ");
				print_time_from_seconds(stdout, total_time);
				fflush(stdout);
				printf("\nStarting stage 1 gcd.\n");
				st1_factor = get_gcd(/*x,*/x_packed, q, n, 1);
			}
			if (!st1_factor) {
				if (stage == 3) {
					printf("Here's where we put the b1 extension calls\n");
					stage = 2;
				}
				if (stage == 2) {
					printf("Starting stage 2.\n");
					stage2(x_int, x_packed, q, n, nrp, maxerr);
					if (!quitting) {
						printf("Starting stage 2 gcd.\n");
						get_gcd(x_packed, q, n, 2);
						rm_checkpoint(q, keep_s1);
					}
				}
			}
			printf("\n");
		}
		close_lucas(x_int);
	} while (restarting);
	free((char *) x_packed);
	return (0);
}

int dir_exists(const char *path) {
	struct stat info;

	if(stat( path, &info ) != 0)
		return 0;
	else if(info.st_mode & S_IFDIR)
		return 1;
	else
		return 0;
}

int main(int argc, char *argv[]) {
	printf("%s\n", program);
	quitting = 0;
#define THREADS_DFLT 256
#define CHECKPOINT_ITER_DFLT 10000
#define SAVE_FOLDER_DFLT "savefiles"
#define S_F_DFLT 0
#define T_F_DFLT 0
#define K_F_DFLT 0
#define D_F_DFLT 0
#define POLITE_DFLT 1
#define UNUSEDMEM_DFLT 100
#define WORKFILE_DFLT "worktodo.txt"
#define RESULTSFILE_DFLT "results.txt"
	
	/* "Production" opts to be read in from command line or ini file */
	int q = -1;
	int f_f = 0;
	device_number = -1;
	checkpoint_iter = -1;
	threads1 = -1;
	fftlen = -1;
	unused_mem = -1;
	s_f = t_f = d_f = k_f = -1;
	polite_f = polite = -1;
	AID[0] = input_filename[0] = RESULTSFILE[0] = 0; /* First character is null terminator */
	char fft_str[132] = "\0";
	
	/* Non-"production" opts */
	r_f = 0;
	int cufftbench_s, cufftbench_e, cufftbench_d;
	cufftbench_s = cufftbench_e = cufftbench_d = 0;
	
	parse_args(argc, argv, &q, &device_number, &cufftbench_s, &cufftbench_e, &cufftbench_d);

	/* The rest of the args are globals */

	if (file_exists(INIFILE)) {
		if (checkpoint_iter < 1 && !IniGetInt(INIFILE, "CheckpointIterations", &checkpoint_iter, CHECKPOINT_ITER_DFLT))
			/*fprintf(stderr, "Warning: Couldn't parse ini file option CheckpointIterations; using default: %d\n", CHECKPOINT_ITER_DFLT)*/;
		if (threads1 < 1 && !IniGetInt(INIFILE, "Threads", &threads1, THREADS_DFLT))
			fprintf(stderr, "Warning: Couldn't parse ini file option Threads; using default: %d\n", THREADS_DFLT);
		if (s_f < 0 && !IniGetInt(INIFILE, "SaveAllCheckpoints", &s_f, S_F_DFLT))
			/*fprintf(stderr, "Warning: Couldn't parse ini file option SaveAllCheckpoints; using default: off\n")*/;
		if (s_f > 0 && !IniGetStr(INIFILE, "SaveFolder", folder, SAVE_FOLDER_DFLT))
			/*fprintf(stderr, "Warning: Couldn't parse ini file option SaveFolder; using default: \"%s\"\n", SAVE_FOLDER_DFLT)*/;
		if (t_f < 0 && !IniGetInt(INIFILE, "CheckRoundoffAllIterations", &t_f, 0))
			fprintf(stderr, "Warning: Couldn't parse ini file option CheckRoundoffAllIterations; using default: off\n");
		if (!IniGetInt(INIFILE, "KeepStage1SaveFile", &keep_s1, 0))
			keep_s1 = 0;
		if (polite < 0 && !IniGetInt(INIFILE, "Polite", &polite, POLITE_DFLT))
			fprintf(stderr, "Warning: Couldn't parse ini file option Polite; using default: %d\n", POLITE_DFLT);
		if (k_f < 0 && !IniGetInt(INIFILE, "Interactive", &k_f, 0))
			/*fprintf(stderr, "Warning: Couldn't parse ini file option Interactive; using default: off\n")*/;
		if (device_number < 0 && !IniGetInt(INIFILE, "DeviceNumber", &device_number, 0))
			fprintf(stderr, "Warning: Couldn't parse ini file option DeviceNumber; using default: 0\n");
		if (d_f < 0 && !IniGetInt(INIFILE, "PrintDeviceInfo", &d_f, D_F_DFLT))
			/*fprintf(stderr, "Warning: Couldn't parse ini file option PrintDeviceInfo; using default: off\n")*/;
		if (!input_filename[0] && !IniGetStr(INIFILE, "WorkFile", input_filename, WORKFILE_DFLT))
			fprintf(stderr, "Warning: Couldn't parse ini file option WorkFile; using default \"%s\"\n", WORKFILE_DFLT);
		/* I've readded the warnings about worktodo and results due to the multiple-instances-in-one-dir feature. */
		if (!RESULTSFILE[0] && !IniGetStr(INIFILE, "ResultsFile", RESULTSFILE, RESULTSFILE_DFLT))
			fprintf(stderr, "Warning: Couldn't parse ini file option ResultsFile; using default \"%s\"\n", RESULTSFILE_DFLT);
		if (fftlen < 0 && !IniGetStr(INIFILE, "FFTLength", fft_str, "\0"))
			/*fprintf(stderr, "Warning: Couldn't parse ini file option FFTLength; using autoselect.\n")*/;
		if (unused_mem < 0 && !IniGetInt(INIFILE, "UnusedMem", &unused_mem, UNUSEDMEM_DFLT))
			printf("Warning: Couldn't find or parse ini file option UnusedMem; using default %dMiB.\n", UNUSEDMEM_DFLT);
	} else  // no ini file
	{
		fprintf(stderr, "Warning: Couldn't find .ini file. Using defaults for non-specified options.\n");
		if (checkpoint_iter < 1)
			checkpoint_iter = CHECKPOINT_ITER_DFLT;
		if (threads1 < 1)
			threads1 = THREADS_DFLT;
		if (fftlen < 0)
			fftlen = 0;
		if (s_f < 0)
			s_f = S_F_DFLT;
		if (t_f < 0)
			t_f = T_F_DFLT;
		if (k_f < 0)
			k_f = K_F_DFLT;
		if (device_number < 0)
			device_number = 0;
		if (d_f < 0)
			d_f = D_F_DFLT;
		if (polite < 0)
			polite = POLITE_DFLT;
		if (unused_mem < 0)
			unused_mem = UNUSEDMEM_DFLT
		;
		if (!input_filename[0])
			sprintf(input_filename, WORKFILE_DFLT);
		if (!RESULTSFILE[0])
			sprintf(RESULTSFILE, RESULTSFILE_DFLT);
	}
	
	if (fftlen < 0) {  // possible if -f not on command line
		fftlen = fft_from_str(fft_str);
	}
	if (polite == 0) {
		polite_f = 0;
		polite = 1;
	} else {
		polite_f = 1;
	}
	if (threads1 != 32 && threads1 != 64 && threads1 != 128 && threads1 != 256 && threads1 != 512 && threads1 != 1024) {
		fprintf(stderr, "Error: thread count is invalid.\n");
		fprintf(stderr, "Threads must be 2^k, 5 <= k <= 10.\n\n");
		exit(2);
	}
	f_f = fftlen;  // if the user has given an override... then note this length must be kept between tests

	init_device(device_number);
	fft_count = init_ffts();
	
	if (cufftbench_d)
		cufftbench(cufftbench_s, cufftbench_e, cufftbench_d, device_number);
	else {
		if (s_f) {
			if (!dir_exists(folder)) {
#ifndef _MSC_VER
				mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
				if (mkdir(folder, mode) != 0) {
#else
				if (_mkdir (folder) != 0) {
#endif
					fprintf (stderr, "mkdir: cannot create directory `%s': aborting\n", folder);
					exit(EXIT_FAILURE);
				}
			}
		}

		if (selftest)
			run_selftests(selftest);
		else if (q <= 0) {
			int error;
			
#ifdef EBUG
			printf("Processed INI file and console arguments correctly; about to call get_next_assignment().\n");
#endif
			do {  // while(!quitting)
			
				fftlen = f_f;  // fftlen and AID change between tests, so be sure to reset them
				AID[0] = 0;
				
				error = get_next_assignment(input_filename, &q, &fftlen, &tfdepth, &llsaved, &AID, &g_b1_commandline, &g_b2_commandline);
				
				/* Guaranteed to write to fftlen ONLY if specified on workfile line, so that if unspecified, the pre-set default is kept. */
				if (error > 0)
					exit(2);  // get_next_assignment prints warning message
#ifdef EBUG
							printf("Gotten assignment, about to call check().\n");
#endif
				
				check_pm1(q, 0);
				
				if (!quitting)  // Only clear assignment if not killed by user, i.e. test finished
				{
					error = clear_assignment(input_filename, q);
					if (error)
						exit(2);  // prints its own warnings
				}
				
			} while (!quitting);
		} else  // Exponent passed in as argument
		{
			if (!valid_assignment(q, fftlen)) {
				printf("\n");
			}  //! v_a prints warning
			else {  //int trft = 0;
//while(!trft)
				{
					check_pm1(q, 0);
					//q += 2;
					//while(!isprime(q)) q += 2;
				}
			}
		}
	}  // end if(-r) else if(-cufft) else(workfile)
}  // end main()

void parse_args(int argc, char *argv[], int* q, int* device_number, int* cufftbench_s, int* cufftbench_e, int* cufftbench_d) {
	while (argc > 1) {
		if (strcmp(argv[1], "-t") == 0) {
			t_f = 1;
			argv++;
			argc--;
		} else if (strcmp(argv[1], "-h") == 0) {
			fprintf(stderr, "$ CUDAPm1 -h|-v\n\n");
			fprintf(stderr,
					"$ CUDAPm1 [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-c checkpoint_iteration] [-f fft_length] [-s folder] [-t] [-polite iteration] [-k] exponent|input_filename\n\n");
			fprintf(stderr,
					"$ CUDAPm1 [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-polite iteration] -r\n\n");
			fprintf(stderr, "$ CUDAPm1 [-d device_number] [-info] -cufftbench start end distance\n\n");
			fprintf(stderr, "                       -h          print this help message\n");
			fprintf(stderr, "                       -v          print version number\n");
			fprintf(stderr, "                       -info       print device information\n");
			fprintf(stderr, "                       -i          set .ini file name (default = \"CUDAPm1.ini\")\n");
			fprintf(stderr, "                       -threads    set threads number (default = 256)\n");
			fprintf(stderr, "                       -f          set fft length (if round off error then exit)\n");
			fprintf(stderr, "                       -s          save all checkpoint files\n");
			fprintf(stderr, "                       -t          check round off error all iterations\n");
			fprintf(stderr,
					"                       -polite     GPU is polite every n iterations (default -polite 1) (-polite 0 = GPU aggressive)\n");
			fprintf(stderr,
					"                       -cufftbench exec CUFFT benchmark (Ex. $ ./CUDAPm1 -d 1 -cufftbench 1179648 6291456 32768 )\n");
			fprintf(stderr, "                       -r          exec residue test.\n");
			fprintf(stderr, "                       -k          enable keys (p change -polite, t disable -t, s change -s)\n\n");
			fprintf(stderr, "                       -b2         set b2\n\n");
			fprintf(stderr, "                       -d2         Brent-Suyama coefficient (multiple of 30, 210, or 2310) \n\n");
			fprintf(stderr, "                       -e2         Brent-Suyama exponent (2-12) \n\n");
			fprintf(stderr, "                       -selftest   Run a quick selftest (ETA: ");
			print_time_from_seconds(stderr, summarise_selftests(1));
			fprintf(stderr, ")\n");
			fprintf(stderr, "                       -selftest2  Run a longer selftest (ETA: ");
			print_time_from_seconds(stderr, summarise_selftests(2));
			fprintf(stderr, ")\n\n");
			exit(2);
		} else if (strcmp(argv[1], "-v") == 0) {
			printf("%s\n\n", program);
			exit(2);
		} else if (strcmp(argv[1], "-polite") == 0) {
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -polite option\n\n");
				exit(2);
			}
			polite = atoi(argv[2]);
			if (polite == 0) {
				polite_f = 0;
				polite = 1;
			}
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-r") == 0) {
			r_f = 1;
			argv++;
			argc--;
		} else if (strcmp(argv[1], "-k") == 0) {
			k_f = 1;
			argv++;
			argc--;
		} else if (strcmp(argv[1], "-d") == 0) {
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -d option\n\n");
				exit(2);
			}
			*device_number = atoi(argv[2]);
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-i") == 0) {
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -i option\n\n");
				exit(2);
			}
			sprintf(INIFILE, "%s", argv[2]);
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-info") == 0) {
			d_f = 1;
			argv++;
			argc--;
		} else if (strcmp(argv[1], "-noinfo") == 0) {
			d_f = 0;
			argv++;
			argc--;
		} else if (strcmp(argv[1], "-cufftbench") == 0) {
			if (argc < 5 || argv[2][0] == '-' || argv[3][0] == '-' || argv[4][0] == '-') {
				fprintf(stderr, "can't parse -cufftbench option\n\n");
				exit(2);
			}
			*cufftbench_s = atoi(argv[2]);
			*cufftbench_e = atoi(argv[3]);
			*cufftbench_d = atoi(argv[4]);
			argv += 4;
			argc -= 4;
		} else if (strcmp(argv[1], "-threads") == 0) {
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -threads option\n\n");
				exit(2);
			}
			threads1 = atoi(argv[2]);
			if (threads1 != 32 && threads1 != 64 && threads1 != 128 && threads1 != 256 && threads1 != 512 && threads1 != 1024) {
				fprintf(stderr, "Error: thread count is invalid.\n");
				fprintf(stderr, "Threads must be 2^k, 5 <= k <= 10.\n\n");
				exit(2);
			}
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-c") == 0) {
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -c option\n\n");
				exit(2);
			}
			checkpoint_iter = atoi(argv[2]);
			if (checkpoint_iter == 0) {
				fprintf(stderr, "can't parse -c option\n\n");
				exit(2);
			}
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-f") == 0) {
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -f option\n\n");
				exit(2);
			}
			fftlen = fft_from_str(argv[2]);
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-b1") == 0) {
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -b1 option\n\n");
				exit(2);
			}
			g_b1_commandline = atoi(argv[2]);
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-e2") == 0) {
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -e2 option\n\n");
				exit(2);
			}
			g_e = atoi(argv[2]);
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-d2") == 0) {
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -d2 option\n\n");
				exit(2);
			}
			g_d_commandline = atoi(argv[2]);
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-b2") == 0) {
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -b2 option\n\n");
				exit(2);
			}
			g_b2_commandline = atoi(argv[2]);
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-nrp2") == 0) {
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -nrp option\n\n");
				exit(2);
			}
			g_nrp = atoi(argv[2]);
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-s") == 0) {
			s_f = 1;
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -s option\n\n");
				exit(2);
			}
			sprintf(folder, "%s", argv[2]);
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-eb1") == 0) {
			s_f = 1;
			if (argc < 3 || argv[2][0] == '-') {
				fprintf(stderr, "can't parse -eb1 option\n\n");
				exit(2);
			}
			g_eb1 = atoi(argv[2]);
			argv += 2;
			argc -= 2;
		} else if (strcmp(argv[1], "-selftest") == 0) {
			selftest = 1;
			argv++;
			argc--;
		} else if (strcmp(argv[1], "-selftest2") == 0) {
			selftest = 2;
			argv++;
			argc--;
		} else {
			if (*q != -1 || strcmp(input_filename, "") != 0) {
				fprintf(stderr, "can't parse options\n\n");
				exit(2);
			}
			int derp = atoi(argv[1]);
			if (derp == 0) {
				sprintf(input_filename, "%s", argv[1]);
			} else {
				*q = derp;
				*q |= 1;
				while (!isprime(*q))
					*q += 2;

				// If q specified on cmd line, we aren't given tfdepth or llsaved.
				// Estimate based on P95 defaults
				tfdepth = ceil(log(*q) * 3.95);
				printf("Assuming exponent is trial factored to %d bits\n", tfdepth);
				
				// We know all exponents below this have been LL'd at least once.
				// 2014-03-25
				if ((*q) <= 50000000)
					llsaved = 1;

			}
			argv++;
			argc--;
		}
	}
	if (g_d_commandline % 30 != 0) {
		printf("-d2 must be a multiple of 30, 210, or 2310.\n");
		exit(3);
	}
	if ((g_e % 2 != 0) || (g_e < 0) || (g_e > 12)) {
		printf("-e2 must be 2, 4, 6, 8, 10, or 12.\n");
		exit(3);
	}
}

int interact(void) {
	int c = getchar();
	switch (c) {
		case 'p':
			if (polite_f) {
				polite_f = 0;
				printf("   -polite 0\n");
			} else {
				polite_f = 1;
				printf("   -polite %d\n", polite);
			}
			break;
		case 't':
			t_f = 0;
			printf("   disabling -t\n");
			break;
		case 's':
			if (s_f == 1) {
				s_f = 2;
				printf("   disabling -s\n");
			} else if (s_f == 2) {
				s_f = 1;
				printf("   enabling -s\n");
			}
			break;
		case 'F':
			printf(" -- Increasing fft length.\n");
			fftlen++;
			return 1;
		case 'f':
			printf(" -- Decreasing fft length.\n");
			fftlen--;
			return 1;
		case 'k':
			printf(" -- fft length reset cancelled.\n");
			return 2;
	}
	fflush (stdin);
	return 0;
}
