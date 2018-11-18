/*
 This file used to be part of mfaktc.
 Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)
 This file has been written by Luigi Morelli (L.Morelli@mclink.it)

 mfaktc is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 mfaktc is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _PARSE_H
#define _PARSE_H

#include <stdio.h>

enum ASSIGNMENT_TYPE {
	NONE,
	PFACTOR,
	PMINUS1
};

enum ASSIGNMENT_ERRORS {
	NEVER_ASSIGNED = -1,
	OK = 0,
	CANT_OPEN_FILE = 1,
	VALID_ASSIGNMENT_NOT_FOUND = 2,
	CANT_OPEN_WORKFILE = 3,
	CANT_OPEN_TEMPFILE = 4,
	ASSIGNMENT_NOT_FOUND = 5,
	CANT_RENAME = 6
};

enum PARSE_WARNINGS {
	NO_WARNING = 0,
	END_OF_FILE,
	LONG_LINE,
	NO_TEST_EQUAL,
	INVALID_FORMAT,
	INVALID_DATA,
	BLANK_LINE,
	NONBLANK_LINE
};

#define MAX_LINE_LENGTH 131
typedef char LINE_BUFFER[MAX_LINE_LENGTH + 1];

struct ASSIGNMENT {
	enum ASSIGNMENT_TYPE type;
	int exponent;
	int fft_length;
	int tf_depth;
	int ll_saved;
	int b1;
	int b2;
	char hex_key[MAX_LINE_LENGTH + 1];	// optional assignment key....
	char comment[MAX_LINE_LENGTH + 1];	// optional comment -- it followed the assignment on the same line.
};

#ifdef __CUDACC__
/* We must declare these as such because CUDA is recognized as C++, a different language. See:
 http://forums.nvidia.com/index.php?showtopic=190973 */
extern "C" enum ASSIGNMENT_ERRORS get_next_assignment(char *filename, int *exponent, int* fft_length, int* tf_depth, int* ll_saved,
		LINE_BUFFER *assignment_key, int *b1, int *b2);
extern "C" enum ASSIGNMENT_ERRORS clear_assignment(char *filename, int exponent);
extern "C" int valid_assignment(int exp, int fft_length);	// nonzero if assignment is not horribly invalid
extern "C" int IniGetInt(const char* ini_file, const char* name, int* value, int dfault);
extern "C" int IniGetStr(const char* ini_file, const char* name, char* str, const char* dfault);
extern "C" int file_exists(char* filename);  // nonzero if file exists
extern "C" FILE* fopen_and_lock(const char *path, const char *mode);
extern "C" int unlock_and_fclose(FILE* f);
extern "C" int isprime(unsigned int n);
extern "C" void remove_spaces (char* str_trimmed, const char* str_untrimmed);
#else /* __CUDACC__ */
enum ASSIGNMENT_ERRORS get_next_assignment(char *filename, int *exponent, int* fft_length, int* tf_depth, int* ll_saved,
		LINE_BUFFER *assignment_key, int *b1, int *b2);
enum ASSIGNMENT_ERRORS clear_assignment(char *filename, int exponent);
int valid_assignment(int exp, int fft_length);	// nonzero if assignment is not horribly invalid
int IniGetInt(const char* ini_file, const char* name, int* value, int dfault);
int IniGetStr(const char* ini_file, const char* name, char* str, const char* dfault);
int file_exists(char* filename);  // nonzero if file exists
FILE* fopen_and_lock(const char *path, const char *mode);
int unlock_and_fclose(FILE* f);
int isprime(unsigned int n);
void remove_spaces (char* str_trimmed, const char* str_untrimmed);
#endif /* __CUDACC__ */

/* In order to have the gettimeofday() function, you need
 * these includes on Linux:
 * 	#include <sys/time.h>
 * 	#include <unistd.h>
 * On Windows, you need
 * 	#include <winsock2.h>
 * and a definition for
 * 	int gettimeofday (struct timeval *tv, struct timezone) {}
 */

#ifndef _MSC_VER
#include <sys/time.h>
#include <unistd.h>
#else
#include <winsock2.h>

#ifdef __CUDACC__
extern "C" int gettimeofday (struct timeval *tv, struct timezone *);
#else /* __CUDAACC__ */
int gettimeofday (struct timeval *tv, struct timezone *);
#endif /* __CUDACC__ */
#endif

#endif /* _PARSE_H */
