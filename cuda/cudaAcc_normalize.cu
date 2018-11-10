#include "cuda_functions.h"

template <int g_err_flag, int bench>
__global__ void norm1a_kernel(double *g_in, int *g_data, int *g_xint, double *g_ttmp, int *g_carry, volatile float *g_err, float maxerr) {
	long long int bigint[2];
	int val[2], numbits[2] = {g_qn[0], g_qn[0]}, mask[2], shifted_carry, carry_tmp, err_tmp;
	double ttp_temp;
	const int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
	const int index1 = blockIdx.x << 1;
	__shared__ int carry[1024 + 1];

	double tval[2], trint[2];
	float ferr[2];

	tval[0] = g_ttmp[index];
	ttp_temp = g_ttmp[index + 1];
	trint[0] = g_in[index];
	trint[1] = g_in[index + 1];
	if (tval[0] < 0.0) {
		numbits[0]++;
		tval[0] = -tval[0];
	}
	if (ttp_temp < 0.0) {
		numbits[1]++;
		ttp_temp = -ttp_temp;
	}
	tval[1] = tval[0] * g_ttp_inc[numbits[0] == g_qn[0]];
	tval[0] = trint[0] * tval[0];
	tval[1] = trint[1] * tval[1];
	trint[0] = RINT(tval[0]);
	ferr[0] = fabs(tval[0] - trint[0]);
	bigint[0] = (long long int) trint[0];
	trint[1] = RINT(tval[1]);
	ferr[1] = fabs(tval[1] - trint[1]);
	bigint[1] = (long long int) trint[1];
	mask[0] = -1 << numbits[0];
	mask[1] = -1 << numbits[1];
	ferr[0] = fmax(ferr[0], ferr[1]);

	err_tmp = __float_as_int(ferr[0]);

	if (!bench) {
		if (ferr[0] > maxerr)
			atomicMax((int*) g_err, err_tmp);
	}

	val[1] = ((int) bigint[1]) & ~mask[1];
	carry[threadIdx.x + 1] = (int) (bigint[1] >> numbits[1]);
	__syncthreads();

	carry_tmp = carry[threadIdx.x];
	val[0] = ((int) bigint[0]) & ~mask[0];
	val[1] += (int) (bigint[0] >> numbits[0]);

	if (threadIdx.x)
		val[0] += carry_tmp;
	shifted_carry = val[1] - (mask[1] >> 1);
	val[1] = val[1] - (shifted_carry & mask[1]);
	carry[threadIdx.x] = shifted_carry >> numbits[1];
	__syncthreads();

	carry_tmp = carry[threadIdx.x + 1] + carry[threadIdx.x];
	shifted_carry = val[0] - (mask[0] >> 1);
	val[0] = val[0] - (shifted_carry & mask[0]);
	val[1] += shifted_carry >> numbits[0];

	if (threadIdx.x == (blockDim.x - 1)) {
		if (blockIdx.x == gridDim.x - 1)
			g_carry[0] = carry_tmp;
		else
			g_carry[blockIdx.x + 1] = carry_tmp;
	}

	if (threadIdx.x) {
		val[0] += carry[threadIdx.x - 1];

		g_in[index + 1] = (double) val[1] * ttp_temp;
		ttp_temp *= -g_ttp_inc[numbits[0] == g_qn[0]];
		g_in[index] = (double) val[0] * ttp_temp;
		if (g_err_flag) {
			g_xint[index + 1] = val[1];
			g_xint[index] = val[0];
		}
	} else {
		g_data[index1] = val[0];
		g_data[index1 + 1] = val[1];
	}
}

template <int g_err_flag>
__global__ void norm1b_kernel(double *g_in, long long int *g_data, int *g_xint, double *g_ttmp, long long int *g_carry, volatile float *g_err,
		float maxerr) {
	long long int bigint[2], shifted_carry, carry_tmp;
	int numbits[2] = {g_qn[0], g_qn[0]}, mask[2];
	double ttp_temp;
	const int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
	const int index1 = blockIdx.x << 1;
	__shared__ long long int carry[1024 + 1];

	double tval[2], trint[2];
	float ferr[2];

	tval[0] = g_ttmp[index];
	ttp_temp = g_ttmp[index + 1];
	trint[0] = g_in[index];
	trint[1] = g_in[index + 1];
	if (tval[0] < 0.0) {
		numbits[0]++;
		tval[0] = -tval[0];
	}
	if (ttp_temp < 0.0) {
		numbits[1]++;
		ttp_temp = -ttp_temp;
	}
	tval[1] = tval[0] * g_ttp_inc[numbits[0] == g_qn[0]];
	tval[0] = trint[0] * tval[0];
	tval[1] = trint[1] * tval[1];
	trint[0] = RINT(tval[0]);
	ferr[0] = fabs(tval[0] - trint[0]);
	bigint[0] = (long long int) trint[0];
	trint[1] = RINT(tval[1]);
	ferr[1] = fabs(tval[1] - trint[1]);
	bigint[1] = (long long int) trint[1];
	mask[0] = -1 << numbits[0];
	mask[1] = -1 << numbits[1];
	ferr[0] = fmax(ferr[0], ferr[1]);

	if (ferr[0] > maxerr) {
		atomicMax((int*) g_err, __float_as_int(ferr[0]));
	}
	bigint[0] *= 3;
	bigint[1] *= 3;
	carry[threadIdx.x + 1] = (bigint[1] >> numbits[1]);
	__syncthreads();

	carry_tmp = carry[threadIdx.x];
	bigint[1] = bigint[1] & ~mask[1];
	bigint[1] += bigint[0] >> numbits[0];
	bigint[0] = bigint[0] & ~mask[0];

	if (threadIdx.x) {
		bigint[0] += carry_tmp;
	}
	shifted_carry = bigint[1] - (mask[1] >> 1);
	bigint[1] = bigint[1] - (shifted_carry & mask[1]);
	carry[threadIdx.x] = shifted_carry >> numbits[1];
	__syncthreads();

	carry_tmp = carry[threadIdx.x + 1] + carry[threadIdx.x];
	shifted_carry = bigint[0] - (mask[0] >> 1);
	bigint[0] = bigint[0] - (shifted_carry & mask[0]);
	bigint[1] += shifted_carry >> numbits[0];

	if (threadIdx.x == (blockDim.x - 1)) {
		if (blockIdx.x == gridDim.x - 1) {
			g_carry[0] = carry_tmp;
		} else {
			g_carry[blockIdx.x + 1] = carry_tmp;
		}
	}

	if (threadIdx.x) {
		bigint[0] += carry[threadIdx.x - 1];
		g_in[index + 1] = (double) bigint[1] * ttp_temp;
		ttp_temp *= -g_ttp_inc[numbits[0] == g_qn[0]];
		g_in[index] = (double) bigint[0] * ttp_temp;
		if (g_err_flag) {
			g_xint[index + 1] = bigint[1];
			g_xint[index] = bigint[0];
		}
	} else {
		g_data[index1] = bigint[0];
		g_data[index1 + 1] = bigint[1];
	}
}

template <int g_err_flag>
__global__ void norm2a_kernel(double *g_x, int *g_xint, int g_N, int threads1, int *g_data, int *g_carry, double *g_ttp1) {
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	const int threadID1 = threadID << 1;
	const int j = (threads1 * threadID) << 1;
	int temp0, temp1;
	int mask, shifted_carry, numbits = g_qn[0];
	double temp;

	if (j < g_N) {
		temp0 = g_data[threadID1] + g_carry[threadID];
		temp1 = g_data[threadID1 + 1];
		temp = g_ttp1[threadID];
		if (temp < 0.0) {
			numbits++;
			temp = -temp;
		}
		mask = -1 << numbits;
		shifted_carry = temp0 - (mask >> 1);
		temp0 = temp0 - (shifted_carry & mask);
		temp1 += (shifted_carry >> numbits);
		g_x[j + 1] = temp1 * temp;
		temp *= -g_ttp_inc[numbits == g_qn[0]];
		g_x[j] = temp0 * temp;
		if (g_err_flag) {
			g_xint[j + 1] = temp1;
			g_xint[j] = temp0;
		}
	}
}

template <int g_err_flag>
__global__ void norm2b_kernel(double *g_x, int *g_xint, int g_N, int threads1, long long int *g_data, long long int *g_carry, double *g_ttp1) {

	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	const int threadID1 = threadID << 1;
	const int j = (threads1 * threadID) << 1;
	long long int shifted_carry, temp0, temp1;
	int mask, numbits = g_qn[0];
	double temp;

	if (j < g_N) {
		temp0 = g_data[threadID1] + g_carry[threadID];
		temp1 = g_data[threadID1 + 1];
		temp = g_ttp1[threadID];

		if (temp < 0.0) {
			numbits++;
			temp = -temp;
		}

		mask = -1 << numbits;
		shifted_carry = temp0 - (mask >> 1);
		temp0 = temp0 - (shifted_carry & mask);
		temp1 = temp1 + (shifted_carry >> numbits);
		g_x[j + 1] = temp1 * temp;
		temp *= -g_ttp_inc[numbits == g_qn[0]];
		g_x[j] = temp0 * temp;

		if (g_err_flag) {
			g_xint[j + 1] = temp1;
			g_xint[j] = temp0;
		}
	}
}

__global__ void apply_weights_kernel(double *g_out, int *g_in, double *g_ttmp) {
	int val[2], test = 1;
	double ttp_temp[2];
	const int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;

	val[0] = g_in[index];
	val[1] = g_in[index + 1];
	ttp_temp[0] = g_ttmp[index];
	ttp_temp[1] = fabs(g_ttmp[index + 1]);

	test = ttp_temp[0] < 0.0 ? 0 : 1;

	g_out[index + 1] = (double) val[1] * ttp_temp[1];
	ttp_temp[1] *= -g_ttp_inc[test];
	g_out[index] = (double) val[0] * ttp_temp[1];
}

void set_qn(int *h_qn) {
	cudaMemcpyToSymbol(g_qn, h_qn, 2 * sizeof(int));
}

void set_ttp_inc(double *h_ttp_inc) {
	cudaMemcpyToSymbol(g_ttp_inc, h_ttp_inc, 2 * sizeof(double));
}


void cudaAcc_norm1a(int g_err_flag, int bench, int threads, int n, double *g_in, int *g_data, int *g_xint, double *g_ttmp, int *g_carry, volatile float *g_err, float maxerr) {
	if (bench)
		norm1a_kernel<0,1><<<n / (2 * threads), threads >>>(g_in, g_data, g_xint, g_ttmp, g_carry, g_err, maxerr);
	else if (g_err_flag)
		norm1a_kernel<1,0><<<n / (2 * threads), threads >>>(g_in, g_data, g_xint, g_ttmp, g_carry, g_err, maxerr);
	else
		norm1a_kernel<0,0><<<n / (2 * threads), threads >>>(g_in, g_data, g_xint, g_ttmp, g_carry, g_err, maxerr);
}

void cudaAcc_norm1b(int g_err_flag, int threads, int n, double *g_in, long long int *g_data, int *g_xint, double *g_ttmp, long long int *g_carry, volatile float *g_err, float maxerr) {
	if (g_err_flag)
		norm1b_kernel<1> <<<n / (2 * threads), threads >>> (g_in, g_data, g_xint, g_ttmp, g_carry, g_err, maxerr);
	else
		norm1b_kernel<0> <<<n / (2 * threads), threads >>> (g_in, g_data, g_xint, g_ttmp, g_carry, g_err, maxerr);

}

void cudaAcc_norm2a(int g_err_flag, int threads1, int threads3, double *g_x, int *g_xint, int g_N, int *g_data, int *g_carry, double *g_ttp1) {
	if (g_err_flag)
		norm2a_kernel<1> <<< (g_N / (2 * threads1) + threads3 - 1) / threads3, threads3 >>> (g_x, g_xint, g_N, threads1, g_data, g_carry, g_ttp1);
	else
		norm2a_kernel<0> <<< (g_N / (2 * threads1) + threads3 - 1) / threads3, threads3 >>> (g_x, g_xint, g_N, threads1, g_data, g_carry, g_ttp1);
}

void cudaAcc_norm2b(int g_err_flag, int threads1, int threads3, double *g_x, int *g_xint, int g_N, long long int *g_data, long long int *g_carry, double *g_ttp1) {
	if (g_err_flag)
		norm2b_kernel<1><<< (g_N / (2 * threads1) + threads3 - 1) / threads3, threads3 >>>(g_x, g_xint, g_N, threads1, g_data, g_carry, g_ttp1);
	else
		norm2b_kernel<0><<< (g_N / (2 * threads1) + threads3 - 1) / threads3, threads3 >>>(g_x, g_xint, g_N, threads1, g_data, g_carry, g_ttp1);
}

void cudaAcc_apply_weights(int n, int threads, double *g_out, int *g_in, double *g_ttmp) {
	apply_weights_kernel<<<n / (2 * threads), threads>>>(g_out, g_in, g_ttmp);
}
