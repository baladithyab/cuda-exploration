NVCC := /usr/local/cuda/bin/nvcc
CXX  := clang-14

NVCCFLAGS := -ccbin $(CXX) -O3 -arch=sm_120 -std=c++17 -lstdc++ -lcublas -lm

attn_mla: attn_mla.cu softmax.cu
	$(NVCC) $(NVCCFLAGS) -o $@ attn_mla.cu softmax.cu

clean:
	rm -f attn_mla results.csv

.PHONY: clean
