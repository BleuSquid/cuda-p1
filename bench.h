/*
 * bench.h
 */

#ifndef BENCH_H_
#define BENCH_H_

#include "float.h"
#include "CUDAPm1.h"
#include "cuda/cuda_functions.h"
#include "cuda/cuda_safecalls.h"
#include "parse.h"


void threadbench(int n, int passes, int device_number);
void cufftbench(int cufftbench_s, int cufftbench_e, int passes, int device_number);

#endif /* BENCH_H_ */
