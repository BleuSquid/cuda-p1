NAME = CUDAPm1
VERSION = 0.20 Alpha

BIN = $(NAME)
DEBUG_BIN = $(NAME)_debug

COMMON_INCLUDES =
COMMON_DEFINES =
COMMON_LIBS =
OPTLEVEL = -O3

CXX = g++
CC = gcc
NVCC = nvcc

CFLAGS = $(OPTLEVEL) $(COMMON_INCLUDES) $(COMMON_DEFINES) -Wall
DEBUG_CFLAGS = -g -O0 $(COMMON_INCLUDES) $(COMMON_DEFINES)

# Uncomment the relevant line for your hardware,
# or leave all uncommented for a generic binary
# sm_21 is actually slower than sm_20 on sm_21 hardware...

#NVCC_ARCHES += -gencode arch=compute_13,code=sm_13
#NVCC_ARCHES += -gencode arch=compute_20,code=sm_20
#NVCC_ARCHES += -gencode arch=compute_20,code=sm_21
NVCC_ARCHES += -gencode arch=compute_30,code=sm_30
#NVCC_ARCHES += -gencode arch=compute_35,code=sm_35

# Use --ptxas-options -v to see register usage
# Use --maxrregcount to specify register usage
NVCC_CFLAGS = $(OPTLEVEL) $(COMMON_INCLUDES) $(COMMON_DEFINES) $(NVCC_ARCHES) --compiler-options="$(CFLAGS) -fno-strict-aliasing" -use_fast_math --ptxas-options="-dlcm=cg"
NVCC_DEBUG_CFLAGS = -g -O0 $(COMMON_INCLUDES) $(COMMON_DEFINES) $(NVCC_ARCHES) --compiler-options="$(CFLAGS) -fno-strict-aliasing" -use_fast_math --ptxas-options="-v -dlcm=cg"



#CULIB = $(CUDA)/lib/x86-64-linux-gnu
#CUINC = $(CUDA)/include
#CUFLAGS = -O$(OptLevel) --generate-code arch=compute_13,code=sm_13 --generate-code arch=compute_20,code=sm_20 --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_35,code=sm_35 --compiler-options=-Wall -I$(CUINC)
#NVCC_OPTS= --pre-include gcc_cuda_compat.h

# The nVidia CUDA Toolkit will provide both nvcc and the CUDA libraries. If you
# follow their defaults, the necessary files will be installed in your PATH and
# LDPATH. Otherwise, you'll need to manually insert their paths here.

#CC = gcc
#CFLAGS = -O$(OptLevel) -Wall

#L = -lcufft -lcudart -lm -lgmp
#LDFLAGS = $(CFLAGS) -fPIC -L$(CULIB) $(L)

LIBS = -lcufft -lcudart -lm -lgmp
LDFLAGS = $(COMMON_LDFLAGS) -fPIC -Wl,-O1 -Wl,--as-needed -Wl,--sort-common -Wl,--relax

OBJS = parse.o rho.o CUDAPm1.o

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $^ $(CFLAGS) $(LDFLAGS) $(LIBS) -o $@

$(DEBUG_BIN): $(OBJS)
	$(CXX) $^ $(DEBUG_CFLAGS) $(LDFLAGS) $(LIBS) -o $@

CUDAPm1.o: CUDAPm1.cu parse.h cuda_safecalls.h rho.h complex_math.cu CUDAPm1.h
	$(NVCC) $(NVCC_CFLAGS) -c $<

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $<

clean: clean-test
	rm -f *.o *~
	rm -f $(NAME) debug_$(NAME) test_$(NAME)

debug: $(DEBUG_BIN)

test: test-b1 test-b2

test-b1: $(BIN) clean-test
	./$(BIN) -noinfo 968819 -b1 20000

test-b2: $(BIN) clean-test
	./$(BIN) -noinfo 7990427 -b1 983 -b2 124000

clean-test:
	rm -f *968819* *7990427*


help:
	@echo "\n\"make\"           builds CUDAPm1"
	@echo "\"make clean\"     removes object files"
	@echo "\"make debug\"     creates a debug build"
	@echo "\"make help\"      prints this message"
	@echo "\"make test-b1\"   tests a factor found via a small b1"
	@echo "\"make test-b2\"   tests a factor found via a small b2"
	@echo "\"make test\"      run both tests"
