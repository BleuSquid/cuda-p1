
#ifndef _SELFTEST_H
#define _SELFTEST_H

#include "CUDAPm1.h"
#include "parse.h"

void run_selftests(int mode);
int summarise_selftests(int mode);

extern unsigned long long int st_result;

struct TEST_ASSIGNMENT {
	ASSIGNMENT assignment;
	unsigned long long expected_result;
	float eta;
};

#endif /* _SELFTEST_H */
