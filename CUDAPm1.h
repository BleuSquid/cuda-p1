#ifndef CUDAPM1_H_
#define CUDAPM1_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

int interact(void);
#define RINT_x86(x) (floor(x+0.5))

extern double *g_ttmp, *g_ttp1;
extern double *g_x, *g_ct;
extern int *g_xint;

extern char *size;
extern int threads1, threads2, threads3;
extern float *g_err;
extern int *g_datai, *g_carryi;
extern long long int *g_datal, *g_carryl;
extern cufftHandle plan;
extern cudaDeviceProp dev;
extern int multipliers[250];
extern int polite, polite_f;
extern int fftlen, fft_count;
extern int device_number;

void unpack_bits_int(int *x_int, unsigned *packed_x, int q, int n);
void balance_digits_int(int* x, int q, int n);

#endif /* CUDAPM1_H_ */
