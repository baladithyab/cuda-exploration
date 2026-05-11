# Wave 15 — Nsight tooling verification on WSL2 + RTX 5090 + CUDA 13.2

**Date:** 2026-05-11
**Host:** `DESKTOP-CP4EDJH`, WSL2 (`6.6.114.1-microsoft-standard-WSL2`)
**GPU:** NVIDIA GeForce RTX 5090 (sm_120), driver `596.21` (Windows host, passthrough)
**CUDA:** 13.2 (`/usr/local/cuda-13.2`)

## TL;DR

- ✅ **nsys (Nsight Systems) works end-to-end** — but **only the new one at
  `/usr/local/cuda-13.2/nsight-systems-2025.6.3/bin/nsys` (v2025.6.3)**. The
  apt-installed `/usr/bin/nsys` (v2022.4.2) is unusable: it produces a
  `.qdstrm` and says "Importer error status: The importer binary and its
  dependencies were not found."
- ✅ **`nsys stats` on `.nsys-rep` files works** — kernel summary, CUDA API
  summary, memcpy breakdown, all headless, no GUI required.
- ❌ **ncu (Nsight Compute) is blocked** by `ERR_NVGPUCTRPERM` on both the
  apt `ncu` (2022.4.1) and the modern `/usr/local/cuda-13.2/nsight-compute-2026.1.1/ncu`
  (2026.1.1). Kernel still runs, but **zero metrics collected**.
- ❌ **No sudo password on this host** and **WSL2 has no
  `/proc/driver/nvidia/params`** / `/sys/module/nvidia/parameters/`, so the
  standard `modprobe NVreg_RestrictProfilingToAdminUsers=0` workaround is
  structurally inapplicable — the fix has to happen on the Windows side.
- ⚠️ **nsys captures cuda-oxide binaries only at the Driver API level, and
  misses kernel launches** — only `cuModuleLoad` / `cuCtxSetCurrent` /
  `cuModuleUnload` show up. Kernel timeline is blank for
  `oxide-vecadd-bench` under both default and `--trace=cuda,nvtx`. `cudaEvent`
  timing inside the binary still works; nsys just doesn't *see* the launches.

## Tool inventory (exact paths)

```
/usr/bin/nsys                                                       # 2022.4.2  (broken, skip)
/usr/bin/ncu                                                        # 2022.4.1  (same version as modern one for this test)
/usr/local/cuda-13.2/nsight-systems-2025.6.3/bin/nsys               # 2025.6.3  USE THIS
/usr/local/cuda-13.2/nsight-compute-2026.1.1/ncu                    # 2026.1.1  blocked by perm
```

Recommended shell aliases:

```bash
export NSYS=/usr/local/cuda-13.2/nsight-systems-2025.6.3/bin/nsys
export NCU=/usr/local/cuda-13.2/nsight-compute-2026.1.1/ncu
```

## 1. nsys — what works

### Timeline capture + headless stats, C++ target

```bash
cd /home/codeseys/cuda-exploration/cuda-matmul
$NSYS profile -o /tmp/matmul-v2 --force-overwrite=true --stats=true ./matmul
```

Produces:
- `/tmp/matmul-v2.nsys-rep` (188 KB)
- `/tmp/matmul-v2.sqlite`   (839 KB)

`--stats=true` runs 8 post-processing reports inline:
`osrt_sum`, `cuda_api_sum`, `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`,
`cuda_gpu_mem_size_sum` (nvtx_sum and a couple others are skipped cleanly when
no NVTX ranges are present).

Example captured output for cuda-matmul (N=1024/2048/4096 sweep):

```
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances   Avg (ns)   Name
    100.0       285370494         33  8647590.7   matmul(const float *, const float *, float *, int)

 ** CUDA API Summary:
     50.4        260498006         30  8683266.9   cudaEventSynchronize
     33.1        170846850          3 56948950.0   cudaMalloc
     10.7         55058606          9  6117622.9   cudaMemcpy
     ...
 ** CUDA GPU Memory Ops Time:
     73.2         36644734      3  12214911.3   [CUDA memcpy Device-to-Host]
     26.7         13386530      6   2231088.3   [CUDA memcpy Host-to-Device]
```

### Headless stats from an existing rep

```bash
$NSYS stats --report cuda_gpu_kern_sum --force-export=true /tmp/matmul-v2.nsys-rep
```

Note: on first run against a rep, pass `--force-export=true` to (re)build the
sqlite sidecar. The `.sqlite` is queryable directly with `sqlite3` if you want
custom aggregations. GUI is **not** required.

Available `--report` names can be listed via `$NSYS stats --help`; commonly
useful ones: `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, `cuda_api_sum`,
`cuda_gpu_trace`, `nvtx_sum`, `osrt_sum`.

### Python / cuTile target

```bash
PY=/home/codeseys/cuda-exploration/cutile-vecadd-bench/.venv/bin/python
cd /home/codeseys/cuda-exploration/cutile-vecadd-bench
$NSYS profile -o /tmp/cutile-vecadd-profile --force-overwrite=true $PY main.py
```

Produces `/tmp/cutile-vecadd-profile.nsys-rep` (272 KB). Kernel launches from
cupy + cuda.tile are captured on the timeline (verified — sqlite contains
`CUPTI_ACTIVITY_KIND_KERNEL` rows).

### Old-nsys trap (what NOT to do)

```bash
/usr/bin/nsys profile --stats=true ./matmul  # DO NOT USE
```

Produces only `report1.qdstrm` + message:
> Importer error status: The importer binary and its dependencies were not found.
> Unable to retrieve the importer version: skipping importation of the QDSTRM file.

The 2022-era nsys apt package is missing `QdstrmImporter` on its PATH and also
predates sm_120 awareness. Even if you manually run the importer (it *does*
exist at `/usr/local/cuda-13.2/nsight-systems-2025.6.3/host-linux-x64/QdstrmImporter`),
just use the 2025 nsys directly.

## 2. ncu — blocked

### Exact failure, both versions

```bash
$ /usr/bin/ncu --set basic ./matmul
==PROF== Connected to process ... (/home/codeseys/cuda-exploration/cuda-matmul/matmul)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA
GPU Performance Counters on the target device 0. For instructions on enabling
permissions and to get more information see
https://developer.nvidia.com/ERR_NVGPUCTRPERM
==PROF== Disconnected from process ...
==WARNING== No kernels were profiled.

$ /usr/local/cuda-13.2/nsight-compute-2026.1.1/ncu --set basic ./matmul
# same ERR_NVGPUCTRPERM, same "No kernels were profiled."
```

Adding `--target-processes all` does not help — permission is rejected at
attach time, before any kernel launches.

### Why the usual Linux workaround does not apply

On a native Linux box, root does:

```bash
sudo modprobe -r nvidia                                     # or reboot
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
# persist via /etc/modprobe.d/nvidia-profiler.conf
```

On this WSL2 host:

- `sudo -n true` → `sudo: a password is required` (no passwordless sudo).
- `/proc/driver/nvidia/params` does not exist.
- `/sys/module/nvidia/parameters/` does not exist.
- There is no loadable `nvidia.ko` in `/lib/modules/$(uname -r)/` — the CUDA
  driver inside WSL2 is `libcuda.so` over the `dxg` shim to the Windows host
  driver (596.21). There is no Linux-side kernel module to reconfigure.

Therefore the permission has to be flipped **on the Windows host**, not in WSL2.
NVIDIA's documented Windows route:

- NVIDIA Control Panel → Manage GPU Performance Counters → **Allow access to
  the GPU performance counters to all users** (requires admin on Windows and
  restarts the driver).

This is not something we can do from inside WSL2 without out-of-band access to
the Windows desktop. **Until that flag is flipped on the host, ncu is a no-go
for any measurement-intensive work on this box.**

## 3. cuda-oxide + nsys — partial coverage caveat

```bash
cd /home/codeseys/cuda-exploration/oxide-vecadd-bench
export CUDA_HOME=/usr/local/cuda LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so \
       PATH=/usr/lib/llvm-21/bin:$PATH:$HOME/.cargo/bin
$NSYS profile -o /tmp/oxide-vecadd-profile --force-overwrite=true --stats=true \
    target/release/oxide-vecadd-bench
```

Binary runs to completion; nsys captures **only**:

```
 ** CUDA API Summary:
     85.6          1589863          1  1589863.0   cuModuleLoad
     13.3           247463          1   247463.0   cuCtxSetCurrent
      1.1            19915          1    19915.0   cuModuleUnload

SKIPPED: /tmp/oxide-vecadd-profile.sqlite does not contain CUDA kernel data.
SKIPPED: /tmp/oxide-vecadd-profile.sqlite does not contain GPU memory data.
```

No `cuLaunchKernel` rows, no kernel activity. Re-running with
`--trace=cuda,nvtx` does not change this. The cuda-oxide launch path is
apparently not being intercepted by the nsys CUPTI callbacks on this stack
(consistent with nsys seeing the Runtime-API matmul binary cleanly but the
Driver-API-only cuda-oxide path partially).

Workaround for cuda-oxide kernels: **rely on in-binary `cudaEvent` timing**
(Wave 1/3/12 methodology), or add NVTX ranges + re-run under `nsys profile
--trace=cuda,nvtx` to at least pin kernels on the timeline by name — though
even NVTX requires the kernel activity to be captured for overlap analysis.
Worth filing as an open question before depending on nsys for any
cuda-oxide-side attention work.

## 4. Recommended profiling workflow for Wave 16+ attention kernels

### Timeline + kernel summary (works today, use this for Wave 16)

```bash
export NSYS=/usr/local/cuda-13.2/nsight-systems-2025.6.3/bin/nsys
$NSYS profile -o /tmp/attn-bench --force-overwrite=true --stats=true ./bench
$NSYS stats --report cuda_gpu_kern_sum --force-export=true /tmp/attn-bench.nsys-rep
$NSYS stats --report cuda_gpu_mem_time_sum /tmp/attn-bench.nsys-rep
```

What you get: per-kernel total/avg/min/max ns, instance count, memcpy vs
kernel breakdown, H2D/D2H byte totals. Good enough for: kernel-launch
overhead, relative kernel timing, memory traffic characterization,
flame-graph-style drill-downs in the GUI if opened on a Windows/Linux
desktop.

What you do NOT get: occupancy, SM utilization, memory-throughput ceiling,
instruction mix, stall reasons — those are ncu territory.

### Per-kernel metrics (blocked — document and move on)

```bash
# Would work on unrestricted Linux:
$NCU --set basic --kernel-name ::attention_fwd --launch-skip 2 --launch-count 1 ./bench
$NCU --set full  --kernel-name ::attention_fwd --launch-count 1          ./bench
```

**Blocked on this host** until NVIDIA Control Panel on the Windows side is set
to "Allow access to the GPU performance counters to all users" and the driver
is restarted.

## 5. What's blocked — honest summary

1. **No ncu metrics at all** (occupancy, memory throughput, warp stalls,
   register pressure) without Windows-side profiling permission change.
2. **No cuda-oxide kernel timeline** in nsys (only Driver-API calls show up),
   so Wave 16 attention work routed through cuda-oxide will have to lean on
   `cudaEvent` timing and/or NVTX-annotated runs, not nsys-only analysis.
3. **No GPU clock locking** (`nvidia-smi -lgc`) on WSL2 without Windows admin,
   so per-iter variance continues to be 5–15% at N≥4096 on compute-bound
   kernels (already documented in AGENTS.md from Wave 1).
4. **Old apt nsys (v2022.4.2) is broken** on this box (missing QdstrmImporter,
   no sm_120 support). Always use `/usr/local/cuda-13.2/nsight-systems-2025.6.3/bin/nsys`.
5. **ncu UI / `ncu-ui`** requires an X server; for headless WSL2 it's not
   useful even if profiling worked. `nsys-ui` has the same constraint — use
   `nsys stats` + direct `sqlite3` queries on the `.nsys-rep`-derived sqlite
   for all headless analysis.

## Output file format (reference)

- `.nsys-rep` — current format (2025.6.3 era). Produced directly by `nsys profile`.
- `.qdstrm`   — raw capture, needs QdstrmImporter to become `.nsys-rep`. Only
  seen here from the stale 2022 nsys.
- `.qdrep`    — older "Qdrep" format, superseded by `.nsys-rep`. Not produced
  here.
- `.sqlite`   — auto-derived from `.nsys-rep` via `nsys export` /
  `nsys stats --force-export=true`. Directly queryable, e.g.:

  ```bash
  sqlite3 /tmp/matmul-v2.sqlite \
      "SELECT name, COUNT(*), AVG(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY name;"
  ```

## Commands that were run, exact text

```
which nsys ncu nvidia-smi            # confirms tool paths
nsys --version                       # 2022.4.2  (apt)
ncu --version                        # 2022.4.1  (apt)
/usr/local/cuda-13.2/nsight-systems-2025.6.3/bin/nsys --version    # 2025.6.3
/usr/local/cuda-13.2/nsight-compute-2026.1.1/ncu --version         # 2026.1.1
cat /proc/driver/nvidia/params 2>/dev/null                         # empty — no nvidia proc tree in WSL2
ls /sys/module/nvidia/parameters/ 2>/dev/null                      # empty
sudo -n true                                                        # password required
```

All exact error text quoted above is verbatim from this session, not paraphrased.
