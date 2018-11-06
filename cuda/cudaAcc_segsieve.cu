#include "cuda_functions.h"

__global__ static void segsieve_kernel(uint32 *primes, int maxp, int nump, uint32 N, uint8 *results) {
	/*
	 expect as input a set of primes to sieve with, how many of those primes there are (maxp)
	 how many primes each thread will be responsible for (nump), and the maximum index
	 that we need to worry about for the requested sieve interval.  Also, an array into
	 which we can put this block's count of primes.

	 This routine implements a segmented sieve using a wheel mod 6.  Each thread block on the gpu
	 sieves a different segment of the number line.  Each thread within each block simultaneously
	 sieves a small set of primes, marking composites within shared memory.  There is no memory
	 contention between threads because the marking process is write only.  Because each thread
	 block starts at a different part of the number line, a small amount of computation must
	 be done for each prime prior to sieving to figure out where to start.  After sieving
	 is done, each thread counts primes in part of the shared memory space; the final count
	 is returned in the provided array for each block.  The host cpu will do the final sum
	 over blocks.  Note, it would not be much more difficult to compute and return the primes
	 in the block instead of just the count, but it would be slower due to the extra
	 memory transfer required.
	 */

	uint32 i, j, k;
	uint32 maxID = (N + 1) / 3;
	uint32 bid = blockIdx.y * gridDim.x + blockIdx.x;
	uint32 range = block_size / threadsPerBlock;
	__shared__ uint8 locsieve[block_size];
	__shared__ uint32 bitsieve[block_size / 32];

// everyone init the array.
	if ((bid + 1) * block_size > maxID) {
		for (j = threadIdx.x * range, k = 0; k < range; k++) {
			// we're counting hits in the kernel as well, so clear the bytes representing primes > N
			if ((bid * block_size + j + k) < maxID)
				locsieve[j + k] = 1;
			else
				locsieve[j + k] = 0;
		}
	} else {
		for (j = threadIdx.x * range / 4, k = 0; k < range / 4; k++) {
			((uint32 *) locsieve)[j + k] = 0x01010101;
		}
	}

// the smallest primes are dealt with a bit differently.  They are sieved in a separate
// shared memory space in a packed bit array.  constant memory holds pre-computed
// information about where each prime lands within a given 32 bit region.  each thread
// in the block will use this info to simultaneously sieve a small portion of the
// packed bit array (that way we make use of the broadcast capabilities of constant memory).
// When counting or computing primes, we then have to check both the packed bit array as
// well as the regular byte array, but overall it is a win to greatly speed up the
// sieving of the smallest primes.

// compute starting offset for prime 5:
	i = (bid * 256 + threadIdx.x) % 5;
// then sieve prime 5 in the bit array
	bitsieve[threadIdx.x] = _step5[i];

// compute starting offset for prime 7:
	i = (bid * 256 + threadIdx.x) % 7;
// then sieve prime 7 in the bit array
	bitsieve[threadIdx.x] |= _step7[i];

// compute starting offset for prime 11:
	i = (bid * 256 + threadIdx.x) % 11;
// then sieve prime 11 in the bit array
	bitsieve[threadIdx.x] |= _step11[i];

// compute starting offset for prime 13:
	i = (bid * 256 + threadIdx.x) % 13;
// then sieve prime 13 in the bit array
	bitsieve[threadIdx.x] |= _step13[i];

// compute starting offset for prime 17:
	i = (bid * 256 + threadIdx.x) % 17;
// then sieve prime 17 in the bit array
	bitsieve[threadIdx.x] |= _step17[i];

// compute starting offset for prime 19:
	i = (bid * 256 + threadIdx.x) % 19;
// then sieve prime 19 in the bit array
	bitsieve[threadIdx.x] |= _step19[i];

// regroup before sieving
	__syncthreads();

// now sieve the array
	for (j = 0; j < nump; j++) {
		int pid = (j * threadsPerBlock) + threadIdx.x + startprime;

		if (pid < maxp) {
			uint32 p = primes[pid];
			uint32 pstart = p / 3;
			uint32 p2 = 2 * p;
			uint32 block_start = bid * block_size;
			uint32 start_offset;
			uint32 s[2];

			// the wheel sieve with all multiples of 2 and 3 removed from the array is equivalent to
			// alternately stepping through the number line by (p+2)*mult, (p-2)*mult,
			// where mult = (p+1)/6
			s[0] = p + (2 * ((p + 1) / 6));
			s[1] = p - (2 * ((p + 1) / 6));

			// compute the starting location of this prime in this block
			if ((bid == 0) || (pstart >= block_start)) {
				// start one increment past the starting value of p/3, since
				// we want to count the prime itself as a prime.
				start_offset = pstart + s[0] - block_start;
				k = 1;
			} else {
				// measure how far the start of this block is from where the prime first landed,
				// as well as how many complete (+2/-2) steps it would need to take
				// to cover that distance
				uint32 dist = (block_start - pstart);
				uint32 steps = dist / p2;

				if ((dist % p2) == 0) {
					// if the number of steps is exact, then we hit the start
					// of this block exactly, and we start below with the +2 step.
					start_offset = 0;
					k = 0;
				} else {
					uint32 inc = pstart + steps * p2 + s[0];
					if (inc >= block_start) {
						// if the prime reaches into this block on the first stride,
						// then start below with the -2 step
						start_offset = inc - block_start;
						k = 1;
					} else {
						// we need both +2 and -2 strides to get into the block,
						// so start below with the +2 stride.
						start_offset = inc + s[1] - block_start;
						k = 0;
					}
				}
			}

			// unroll the loop for the smallest primes.
			if (p < 1024) {
				uint32 stop = block_size - (2 * p * 4);

				if (k == 0) {
					for (i = start_offset; i < stop; i += 8 * p) {
						locsieve[i] = 0;
						locsieve[i + s[0]] = 0;
						locsieve[i + p2] = 0;
						locsieve[i + p2 + s[0]] = 0;
						locsieve[i + 4 * p] = 0;
						locsieve[i + 4 * p + s[0]] = 0;
						locsieve[i + 6 * p] = 0;
						locsieve[i + 6 * p + s[0]] = 0;
					}
				} else {
					for (i = start_offset; i < stop; i += 8 * p) {
						locsieve[i] = 0;
						locsieve[i + s[1]] = 0;
						locsieve[i + p2] = 0;
						locsieve[i + p2 + s[1]] = 0;
						locsieve[i + 4 * p] = 0;
						locsieve[i + 4 * p + s[1]] = 0;
						locsieve[i + 6 * p] = 0;
						locsieve[i + 6 * p + s[1]] = 0;
					}
				}
			} else
				i = start_offset;

			// alternate stepping between the large and small strides this prime takes.
			for (; i < block_size; k = !k) {
				locsieve[i] = 0;
				i += s[k];
			}
		}
	}

// regroup before counting
	__syncthreads();

	for (j = threadIdx.x * range, k = 0; k < range; k++)
		locsieve[j + k] = (locsieve[j + k] & ((bitsieve[(j + k) >> 5] & (1 << ((j + k) & 31))) == 0));

	__syncthreads();

	if (threadIdx.x == 0)
		for (k = 0; k < block_size; k++) {
			j = ((bid * block_size + k) * 3 + 1) >> 1;
			if (j < N >> 1)
				results[j] = locsieve[k];
		}
}

void cudaAcc_SegSieve(dim3 grid, int threads, uint32 *primes, int maxp, int nump, uint32 N, uint8 *results) {
	segsieve_kernel<<<grid, threads, 0>>>(primes, maxp, nump, N, results);
}
