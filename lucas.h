/*
 * lucas.h
 */

#ifndef LUCAS_H_
#define LUCAS_H_

float lucas_square(/*double *x,*/int q, int n, int iter, int last, float* maxerr, int error_flag, int bit, int stage, int chkpt);
int* init_lucas_packed_int(unsigned * x_packed, int q, int *n, int *j, int *stage, int *total_time);

#endif /* LUCAS_H_ */
