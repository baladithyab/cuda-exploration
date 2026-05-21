NVCC = /usr/local/cuda/bin/nvcc
CXX  = clang-14

# -O3 + sm_120 native (RTX 5090 Blackwell). lineinfo helps SASS-level inspection.
# --extended-lambda not needed; cuda::pipeline works in a regular __global__ kernel.
NVCCFLAGS = -O3 -arch=sm_120 -ccbin $(CXX) -std=c++17 -lineinfo -lstdc++ -lm

all: attn_gdn_async bench

attn_gdn_async: attn_gdn_async.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bench: bench.cu attn_gdn_async.cu
	$(NVCC) $(NVCCFLAGS) -o $@ bench.cu

# SASS dump — extract instruction-mix evidence per ADR-0004.
sass: attn_gdn_async
	/usr/local/cuda/bin/cuobjdump --dump-sass attn_gdn_async > attn_gdn_async.sass 2>&1 || true
	@echo "----- SASS instruction mix -----"
	@HMMA=$$(grep -c "HMMA"      attn_gdn_async.sass || true); echo "HMMA       : $$HMMA"
	@FFMA=$$(grep -c "FFMA"      attn_gdn_async.sass || true); echo "FFMA       : $$FFMA"
	@LDGE=$$(grep -c "LDG.E"     attn_gdn_async.sass || true); echo "LDG.E      : $$LDGE"
	@LDG128=$$(grep -c "LDG.E.128" attn_gdn_async.sass || true); echo "LDG.E.128  : $$LDG128"
	@LDG64=$$(grep -c "LDG.E.64"  attn_gdn_async.sass || true); echo "LDG.E.64   : $$LDG64"
	@STG128=$$(grep -c "STG.E.128" attn_gdn_async.sass || true); echo "STG.E.128  : $$STG128"
	@MUFU=$$(grep -c "MUFU"      attn_gdn_async.sass || true); echo "MUFU       : $$MUFU"
	@LDGSTS=$$(grep -c "LDGSTS"  attn_gdn_async.sass || true); echo "LDGSTS     : $$LDGSTS  (cp.async — the W22.9 win-instruction)"
	@MBAR=$$(grep -cE "ARRIVES|BAR.ARV|MBAR" attn_gdn_async.sass || true); echo "MBAR.*     : $$MBAR"
	@SYNCS=$$(grep -c "SYNCS"    attn_gdn_async.sass || true); echo "SYNCS      : $$SYNCS"

clean:
	rm -f attn_gdn_async bench attn_gdn_async.sass results.csv run.log build.log

.PHONY: all clean sass
