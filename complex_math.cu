__device__ static void complex_mult(double* a, double* b, double* c) {
	// Multiply two complex numbers a and b, return the results in c.
	// First element is real, second element is imaginary

	c[0] = a[0] * b[0] - a[1] * b[1];
	c[1] = a[0] * b[1] + a[1] * b[0];
}

__device__ static void complex_mult(double* a, double* b) {
	// Multiply two complex numbers a and b, return the results in a.
	// First element is real, second element is imaginary

	double tmp[2];
	complex_mult(a, b, tmp);
	a = tmp;
}

__device__ static void complex_square(double* a, double* b) {
	// Square a complex number a, return the results in b.
	// First element is real, second element is imaginary.

	b[0] = a[0] * a[0] - a[1] * a[1];
	b[1] = 2 * a[0] * a[1];
}

__device__ static void complex_square(double* a) {
	// Square a complex number a, return the results in a.
	// First element is real, second element is imaginary.

	double tmp[2];
	complex_square(a, tmp);
	a = tmp;
}
