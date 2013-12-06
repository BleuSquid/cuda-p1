NAME = CUDAPm1
VERSION = 0.20 Alpha
OptLevel = 3
OUT = $(NAME)
CUDA = /usr/local/cuda

CUC = $(CUDA)/bin/nvcc
CULIB = $(CUDA)/lib64
CUINC = $(CUDA)/include
CUFLAGS = -O$(OptLevel) --generate-code arch=compute_13,code=sm_13 --generate-code arch=compute_20,code=sm_20 --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_35,code=sm_35 --compiler-options=-Wall -I$(CUINC)
#NVCC_OPTS= --pre-include gcc_cuda_compat.h

# The nVidia CUDA Toolkit will provide both nvcc and the CUDA libraries. If you
# follow their defaults, the necessary files will be installed in your PATH and
# LDPATH. Otherwise, you'll need to manually insert their paths here.

CC = gcc
CFLAGS = -O$(OptLevel) -Wall

L = -lcufft -lcudart -lm -lgmp
LDFLAGS = $(CFLAGS) -fPIC -L$(CULIB) $(L)

$(NAME): CUDAPm1.o parse.o
	$(CC) $^ $(LDFLAGS) -o $(OUT)

CUDAPm1.o: CUDAPm1.cu parse.h cuda_safecalls.h
	$(CUC) $(NVCC_OPTS) $(CUFLAGS) -c $<

parse.o: parse.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o
cleaner:
	rm -f $(NAME) debug_$(NAME) test_$(NAME)
cleanest: clean cleaner

debug: CFLAGS += -DEBUG -g
debug: CUFLAGS += -DEBUG -g
debug: OptLevel = 0
debug: OUT = debug_$(NAME)
debug: $(NAME)

test: CFLAGS += -DTEST
test: CUFLAGS += -DTEST
test: OUT = test_$(NAME)
test: $(NAME)

help:
	@echo "\n\"make\"           builds CUDAPm1"
	@echo "\"make clean\"     removes object files"
	@echo "\"make cleaner\"   removes executables"
	@echo "\"make cleanest\"  does both clean and cleaner"
	@echo "\"make debug\"     creates a debug build"
	@echo "\"make test\"      creates an experimental build"
	@echo "\"make help\"      prints this message\n"
