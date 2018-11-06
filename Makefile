NAME = CUDAPm1
VERSION = 0.21

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

# Configure this line to specify which cuda architectures (compute capability) to target
# or leave all supported architectures for a bulky binary that supports all
CUDA_ARCHES = 30 32 35 50 52 53 60 61 62 70 75

# Expand CUDA_ARCHES
NVCC_ARCHES += $(foreach CARCH, $(CUDA_ARCHES), -gencode arch=compute_$(CARCH),code=sm_$(CARCH))

# *Always* include PTX for the highest level supported by this version of NVCC, to
# future-proof the binary for new architectures
NVCC_ARCHES += -gencode arch=compute_75,code=compute_75

# Use --ptxas-options -v to see register usage
# Use --maxrregcount to specify register usage

NVCC_COMMON_CFLAGS = -use_fast_math --ptxas-options="-dlcm=cg -v" --maxrregcount=31 $(NVCC_ARCHES) $(COMMON_INCLUDES) $(COMMON_DEFINES)

NVCC_CFLAGS = $(NVCC_COMMON_CFLAGS) $(OPTLEVEL) --compiler-options="$(CFLAGS) -fno-strict-aliasing"
NVCC_DEBUG_CFLAGS = $(NVCC_COMMON_CFLAGS) -g -O0 --compiler-options="$(DEBUG_CFLAGS) -fno-strict-aliasing"


# The nVidia CUDA Toolkit will provide both nvcc and the CUDA libraries. If you
# follow their defaults, the necessary files will be installed in your PATH and
# LDPATH. Otherwise, you'll need to manually insert their paths here.

LIBS = -lcufft -lcudart -lm -lgmp
LDFLAGS = $(COMMON_LDFLAGS) -fPIC -Wl,-O1 -Wl,--as-needed -Wl,--sort-common -Wl,--relax
DEBUG_LDFLAGS = $(COMMON_LDFLAGS) -fPIC

CUDA_SRCS = $(wildcard cuda/*.cu)
CUDA_OBJS = $(patsubst %.cu,%.o, $(CUDA_SRCS))
OBJS = parse.o rho.o lucas.o CUDAPm1.o

debug: NVCC_CFLAGS = $(NVCC_DEBUG_CFLAGS)
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS = $(DEBUG_LDFLAGS)

all: $(BIN)

$(BIN) $(DEBUG_BIN): $(OBJS) $(CUDA_OBJS)
	$(CXX) $^ $(CFLAGS) $(LDFLAGS) $(LIBS) -o $@

lucas.o: lucas.cu lucas.h CUDAPm1.h parse.h
	$(NVCC) $(NVCC_CFLAGS) -c $<

CUDAPm1.o: CUDAPm1.cu parse.h cuda/cuda_functions.h cuda/cuda_safecalls.h rho.h CUDAPm1.h
	$(NVCC) $(NVCC_CFLAGS) -c $<

cuda/%.o: cuda/%.cu cuda/complex_math.h cuda/cuda_functions.h
	$(NVCC) $(NVCC_CFLAGS) -c $< --output-directory cuda/

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $<

clean: clean-test
	rm -f *.o *~ $(OBJS) $(CUDA_OBJS)
	rm -f $(BIN) $(DEBUG_BIN)

debug: $(DEBUG_BIN)

test: test-b1 test-b2

test-b1: $(BIN) clean-test
	./$(BIN) -noinfo 968819 -b1 20000

test-b2: $(BIN) clean-test
	./$(BIN) -noinfo 7990427 -b1 983 -b2 124000

clean-test:
	rm -f *968819* *7990427* results.txt


help:
	@echo "\n\"make\"           builds CUDAPm1"
	@echo "\"make clean\"     removes object files"
	@echo "\"make debug\"     creates a debug build"
	@echo "\"make help\"      prints this message"
	@echo "\"make test-b1\"   tests a factor found via a small b1"
	@echo "\"make test-b2\"   tests a factor found via a small b2"
	@echo "\"make test\"      run both tests"
