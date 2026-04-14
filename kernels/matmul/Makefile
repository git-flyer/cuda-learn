# Makefile for matmul examples
# Usage:
#   make all        — build all three examples
#   make naive      — build naive_matmul
#   make shared     — build shared_memory_matmul
#   make tile       — build thread_tile_matmul
#   make double_buffering — build double_buffering_matmul
#   make histogram_shared — build histogram_shared
#   make clean      — remove all binaries

NVCC    := nvcc
NVCCFLAGS := -O2 -arch=sm_80   # adjust arch for your GPU (e.g. sm_75 for Turing)

TARGETS := naive_matmul shared_memory_matmul thread_tile_matmul double_buffering_matmul histogram_shared

.PHONY: all naive shared tile double_buffering histogram_shared clean

all: $(TARGETS)

naive: naive_matmul
shared: shared_memory_matmul
tile: thread_tile_matmul
double_buffering: double_buffering_matmul
histogram_shared: histogram_shared

naive_matmul: naive_matmul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

shared_memory_matmul: shared_memory_matmul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

thread_tile_matmul: thread_tile_matmul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

double_buffering_matmul:double_buffering_matmul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

histogram_shared: histogram_shared.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
