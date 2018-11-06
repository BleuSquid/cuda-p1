#ifndef _CUDA_CUDA_COMPLEX_MATH_H
#define _CUDA_CUDA_COMPLEX_MATH_H

#include <cuComplex.h>

__host__ __device__ static __inline__ cuDoubleComplex cuCsqr(cuDoubleComplex a) {
	// Square a complex number
	cuDoubleComplex result;
	result = make_cuDoubleComplex((cuCreal(a) * cuCreal(a)) -
							(cuCimag(a) * cuCimag(a)),
							2.0 * cuCreal(a) * cuCimag(a));
	return result;
}

#endif /* _CUDA_CUDA_COMPLEX_MATH_H */
