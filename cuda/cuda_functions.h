#ifndef CUDA_CUDA_FUNCTIONS_H_
#define CUDA_CUDA_FUNCTIONS_H_

#include "complex_math.h"

typedef unsigned char u8;
typedef unsigned int uint32;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef long long unsigned int uint64;

const int threadsPerBlock = 256;
const uint32 block_size = 8192;
const int startprime = 8;

__constant__ uint32 _step5[5]		= {2418280706, 604570176, 151142544, 37785636, 1083188233};
__constant__ uint32 _step7[7]		= {1107363844, 69210240, 2151809288, 134488080, 276840961, 17302560, 537952322};
__constant__ uint32 _step11[11]		= {33816584, 1073774848, 135266336, 132096, 541065345, 528384, 2164261380, 2113536,
										67110928, 8454146, 268443712};
__constant__ uint32 _step13[13]		= {1075838992, 16809984, 262656, 536875016, 8388672, 67239937, 1050624, 2147500064,
										33554688, 268959748, 4202496, 65664, 134218754};
__constant__ uint32 _step17[17]		= {268435488, 1073741952, 512, 2049, 8196, 32784, 131136, 524544, 2098176, 8392704,
										33570816, 134283264, 537133056, 2148532224, 4194304, 16777218, 67108872};
__constant__ uint32 _step19[19]		= {2147483712, 4096, 262176, 16779264, 1073872896, 8388608, 536870928, 1024, 65544,
										4194816, 268468224, 2097152, 134217732, 256, 16386, 1048704, 67117056, 524288,
										33554433};


void cudaAcc_square(int threads, int n, double *a, double *ct);
void cudaAcc_square1(int threads, int n, double *b, double *a, double *ct);
void cudaAcc_mult2(int threads, int n, double *g_out, double *a, double *b, double *ct);
void cudaAcc_mult3(int threads, int n, double *g_out, double *a, double *b, double *ct);
void cudaAcc_sub_mul(int threads, int n, double *g_out, double *a, double *b1, double *b2, double *ct);
void cudaAcc_pre_mul(int threads, int n, double *a, double *ct);
void cudaAcc_norm1a(int g_err_flag, int bench, int threads, int n, double *g_in, int *g_data, int *g_xint, double *g_ttmp, int *g_carry, volatile float *g_err, float maxerr);
void cudaAcc_norm1b(int g_err_flag, int threads, int n, double *g_in, long long int *g_data, int *g_xint, double *g_ttmp, long long int *g_carry, volatile float *g_err, float maxerr);
void cudaAcc_norm2a(int g_err_flag, int threads1, int threads3, double *g_x, int *g_xint, int g_N, int *g_data, int *g_carry, double *g_ttp1);
void cudaAcc_norm2b(int g_err_flag, int threads1, int threads3, double *g_x, int *g_xint, int g_N, long long int *g_data, long long int *g_carry, double *g_ttp1);
void cudaAcc_apply_weights(int n, int threads, double *g_out, int *g_in, double *g_ttmp);
void cudaAcc_copy(int n, int threads, double *save, double *y);
void cudaAcc_SegSieve(dim3 grid, int threads, uint32 *primes, int maxp, int nump, uint32 N, uint8 *results);

void set_qn(int *h_qn);
void set_ttp_inc(double *h_ttp_inc);

__constant__ double g_ttp_inc[2];
__constant__ int g_qn[2];

# define RINT(x)  __rintd(x)

__device__ static double __rintd(double z) {
	double y;

	asm ("cvt.rni.f64.f64 %0, %1;": "=d" (y):"d" (z));
	return (y);
}

#endif /* CUDA_CUDA_FUNCTIONS_H_ */
