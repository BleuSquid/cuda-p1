/*
 * selfttest.c
 *
 *  Created on: Nov 16, 2018
 *      Author: Aaron Haviland
 */

#include "CUDAPm1.h"
#include "parse.h"

struct TEST_ASSIGNMENT {
	ASSIGNMENT assignment;
	unsigned long long expected_result;
};

struct TEST_ASSIGNMENT fast_exponents[5] = {
		{{PMINUS1,	968819,		55296,		0,	0,	20000,	0,			"", "" },	156948679			}, /* B1 */
		{{PMINUS1,	81119,		4096,		0,	0,	300,	0,			"", "" },	5033271713			}, /* B1 */
		{{PMINUS1,	81119,		4096,		0,	0,	50,		847,		"", "" },	5033271713			}, /* B2 */
		{{PMINUS1,	7990427,	442368,		0,	0,	983,	124000,		"", "" },	10509037975912491881}, /* 21-digit result */
		{{PMINUS1,	4444091,	262144,		0,	0,	7,		2557,		"", "" },	2843992382407199	}  /* B2 */
};

struct TEST_ASSIGNMENT slow_exponents[2] = {
		{{PMINUS1,	200003173,	11468800,	0,	0,	47,		229813,		"", "" },	4320552944485007	}, /* Large exponent, B2 */
		{{PMINUS1,	249500501,	14680064,	0,	0,	41,		113467,		"", "" },	11607130072256471	}  /* Large exponent, B2 */
};

int run_selftests(int mode) {
	int testfail = 0;
	/* run fast selftests first */


	if (mode == 2) {
		/* run slow selftests second */
	}

	return(testfail);
}
