#ifndef CUDAPM1_H_
#define CUDAPM1_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

// Check Architecture
#if _WIN64 || __x86_64__
#define ENV64BIT
#else
#define ENV32BIT
#endif

int interact(void);
#define RINT_x86(x) (floor(x+0.5))

#define MAX_B2 1000000000

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

extern char input_filename[132], RESULTSFILE[132];
extern char INIFILE[132];
extern char AID[132];  // Assignment key
extern char s_residue[32];
extern int tfdepth, llsaved;
extern int g_b1_commandline;
extern int g_b2_commandline;
extern int selftest;

void unpack_bits_int(int *x_int, unsigned *packed_x, int q, int n);
void balance_digits_int(int* x, int q, int n);
int check_pm1(int q, char *expectedResidue);
void parse_args(int argc, char *argv[], int* q, int* device_numer, int* cufftbench_s, int* cufftbench_e, int* cufftbench_d);
/* The rest of the opts are global */

#endif /* CUDAPM1_H_ */
