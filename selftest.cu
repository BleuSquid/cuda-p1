/*
 * selfttest.c
 *
 *  Created on: Nov 16, 2018
 *      Author: Aaron Haviland
 */

#include "selftest.h"

unsigned long long int st_result;

/* Expected time (in seconds) for execution from timings on RTX 2070 w/ 8GiB RAM, AMD FX-4350 @ 4.2GHz */
struct TEST_ASSIGNMENT fast_exponents[5] = {
		/* B-1 */
		{{PMINUS1,	968819,		55296,		0,	0,	20000,	0,			"", "" },	156948679			, 3.0f},
		/* B-1 */
		{{PMINUS1,	81119,		4096,		0,	0,	300,	0,			"", "" },	5033271713			, 0.5f},
		/* B-2 */
		{{PMINUS1,	81119,		4096,		0,	0,	50,		847,		"", "" },	5033271713			, 0.6f},
		/* 21-digit factor */
		{{PMINUS1,	7990427,	442368,		0,	0,	983,	124000,		"", "" },	10509037975912491881, 10.9f},
		/* B-2 */
		{{PMINUS1,	4444091,	262144,		0,	0,	7,		2557,		"", "" },	2843992382407199	, 1.6f}
};

struct TEST_ASSIGNMENT slow_exponents[2] = {
		/* 200M Exponent, B-2, long GCD test */
		{{PMINUS1,	200003173,	11468800,	0,	0,	47,		229813,		"", "" },	4320552944485007	, 525.1f},
		/* 249M Exponent, B-2, long GCD test */
		{{PMINUS1,	249500501,	14680064,	0,	0,	41,		113467,		"", "" },	11607130072256471	, 500.7f}
};

int summarise_selftests(int mode) {
	float eta = 0.0f;
	unsigned int i;

	if (mode == 1) {
		for (i = 0; i < sizeof(fast_exponents) / sizeof(TEST_ASSIGNMENT); i++) {
			eta += fast_exponents[i].eta;
		}
	} else if (mode ==2) {
		for (i = 0; i < sizeof(slow_exponents) / sizeof(TEST_ASSIGNMENT); i++) {
			eta += slow_exponents[i].eta;
		}
	}

	return(int(eta));
}
unsigned int selftest_exponent(struct TEST_ASSIGNMENT test_assg) {
	AID[0] = 0;
	fprintf(stderr, "Testing exponent %d\n", test_assg.assignment.exponent);
	fftlen = test_assg.assignment.fft_length;
	int q = test_assg.assignment.exponent;
	st_result = 0;

	if (valid_assignment(q, fftlen) == 0) {
		fprintf(stderr, "Invalid assignment\n");
		return (1); /* invalid assignment */
	}
	if (test_assg.assignment.type == PFACTOR)
		llsaved = test_assg.assignment.ll_saved;
	if (test_assg.assignment.type == PMINUS1) {
		g_b1_commandline = test_assg.assignment.b1;
		g_b2_commandline = test_assg.assignment.b2;
	}

	/* error = get_next_assignment(input_filename, &q, &fftlen, &tfdepth, &llsaved, &AID, &g_b1_commandline, &g_b2_commandline); */
	return(check_pm1(q, 0));

};

void run_selftests(int mode) {
	int testfail = 0;
	int testpass = 0;
	int passfail = 0;
	unsigned int i = 0;


	fprintf(stderr, "Entering selftest mode: %d\n", selftest);

	/* run fast selftests first */
	for (i = 0; i < sizeof(fast_exponents) / sizeof(TEST_ASSIGNMENT); i++) {
		passfail += selftest_exponent(fast_exponents[i]);
		if (st_result != fast_exponents[i].expected_result)
			passfail++;
		if (passfail) {
			fprintf(stderr, "\n==============\nFAILED TEST!!!\n==============\n");
			testfail++;
		} else {
			testpass++;
		}
	}

	if (mode == 2) {
		/* run slow selftests second */
		for (i = 0; i < sizeof(slow_exponents) / sizeof(TEST_ASSIGNMENT); i++) {
			passfail += selftest_exponent(slow_exponents[i]);
			if (st_result != slow_exponents[i].expected_result)
				passfail++;
			if (passfail) {
				fprintf(stderr, "\n==============\nFAILED TEST!!!\n==============\n");
				testfail++;
			} else {
				testpass++;
			}
		}
	}

	fprintf(stderr, "\nPassed tests: %d\n", testpass);
	fprintf(stderr, "Failed tests: %d\n", testfail);
	return;
}
