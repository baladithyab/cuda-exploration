// Wave 17 W1d — cuda-oxide Gated DeltaNet (GDN) single-timestep decode.
//
// f32 throughout, NO tensor cores (cuda-oxide v0.1.0 has no usable TC API
// on consumer Blackwell sm_120). FFMA-only path; the kernel is strongly
// memory-bound (AI ~0.77 flops/byte) so the FMA quality barely matters.
//
// Reference: cutile-attn-gdn/main.py (algorithm), oxide-attn-gqa/src/main.rs
// (template for harness, NPY loader, kernel-launch plumbing).
// PyTorch oracle: pytorch_reference_gdn.py.
//
// Recurrence per (batch b, head h):
//     S_scaled = α · S_in
//     u        = k · S_scaled              shape (d_v,)
//     r        = v - u
//     S_out    = S_scaled + β · k ⊗ r       shape (d_k, d_v)
//     o        = q · S_out                  shape (d_v,)
//
// Grid (B·H, d_v / BLOCK_V), block (D_K, 1, 1) — ONE THREAD PER d_k ROW.
// Each thread:
//   1. Loads its row of S_in[bh, tid, bv*BV : bv*BV+BV]   (BV elems, registers)
//   2. Multiplies by α (scalar) → S_scaled row
//   3. Loads its k_t (one scalar) into shared mem K_SH
//   4. For each j in [0, BV): partial product k_t * S_scaled[t, j],
//      reduce across threads via shared-mem sum into U_SH[j].
//   5. Computes residual r[j] = v[j] - U_SH[j], and writes S_out[t, j] =
//      S_scaled[t, j] + β · k_t · r[j].   (stays in registers)
//   6. Repeats reduction for o[j] = sum_t q_t · S_out[t, j]  → O_SH[j]
//   7. Thread 0..BV stores O_SH[j] to global o, all threads store S_out.
//
// All reductions use shared-mem-tree reduction since D_K (64 or 256) is
// not a power of 2 multiple suitable for warp-shuffle alone (256 = 8 warps).
// We use a simple block-wide reduction in shared memory.
//
// Two kernels — one specialized for D_K=64 (correctness), one for D_K=256
// (bench / Qwen3-Next-decode). Both share BV=32 (= state tile column block).
//
// Uses core::intrinsics::fmuladdf32 for the inner accumulators (Wave 3
// finding: libNVVM's contractor needs this hint for runtime-bounded loops).

#![feature(core_intrinsics)]
#![allow(internal_features)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig, sys};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

const BV: usize = 32;

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────

/// GDN decode kernel for d_k = 64 (correctness shape). Block = 64 threads.
/// Grid: (B*H, d_v / BV).  d_k=64, d_v=64, BV=32 → grid_y = 2.
#[kernel]
pub fn gdn_decode_dk64(
    q: &[f32],          // (B*H, 64)
    k: &[f32],          // (B*H, 64)
    v: &[f32],          // (B*H, 64)
    alpha: &[f32],      // (B*H,)
    beta: &[f32],       // (B*H,)
    s_in: &[f32],       // (B*H * 64, 64)  flattened (B*H*d_k, d_v)
    mut s_out: DisjointSlice<f32>,
    mut o: DisjointSlice<f32>,
    d_k: u32,
    d_v: u32,
) {
    // Per-block shared mem:
    //   K_SH[D_K]  : f32 — broadcast k vector
    //   Q_SH[D_K]  : f32 — broadcast q vector
    //   PROD[D_K]  : f32 — partial products for tree reduction
    //   UVEC[BV]   : f32 — reduced u[j]
    //   OVEC[BV]   : f32 — reduced o[j]
    //   AB[2]      : f32 — alpha, beta scalars
    static mut K_SH: SharedArray<f32, 64> = SharedArray::UNINIT;
    static mut Q_SH: SharedArray<f32, 64> = SharedArray::UNINIT;
    static mut PROD: SharedArray<f32, 64> = SharedArray::UNINIT;
    static mut UVEC: SharedArray<f32, 32> = SharedArray::UNINIT;
    static mut OVEC: SharedArray<f32, 32> = SharedArray::UNINIT;
    static mut AB: SharedArray<f32, 2> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let bh = thread::blockIdx_x() as usize;
    let bv_idx = thread::blockIdx_y() as usize;

    let dk_us = d_k as usize;
    let dv_us = d_v as usize;
    let _ = dk_us; // == 64
    let _ = dv_us;

    // Load α, β (one thread loads each)
    if tid == 0 {
        unsafe { AB[0] = alpha[bh]; }
    }
    if tid == 1 {
        unsafe { AB[1] = beta[bh]; }
    }
    // Load q, k vectors broadcast (each thread loads its own row index)
    let q_t = q[bh * 64 + tid];
    let k_t = k[bh * 64 + tid];
    unsafe {
        K_SH[tid] = k_t;
        Q_SH[tid] = q_t;
    }
    thread::sync_threads();
    let alpha_v = unsafe { AB[0] };
    let beta_v = unsafe { AB[1] };

    // Load S_in row (this thread's row, BV columns) and scale by α.
    // S_in layout: row-major, row index = bh * d_k + tid; column = bv_idx*BV + j.
    let s_row_base = (bh * 64 + tid) * 64; // d_v=64
    let bv_col0 = bv_idx * BV;
    let mut s_scaled: [f32; BV] = [0.0; BV];
    let mut j = 0usize;
    while j < BV {
        let s = s_in[s_row_base + bv_col0 + j];
        s_scaled[j] = s * alpha_v;
        j += 1;
    }

    // ── Reduction 1: u[j] = sum_t k_t * s_scaled[t, j] ──
    // For each j in [0, BV): each thread writes its product to PROD,
    // then a tree reduce sums PROD[0..D_K] and writes to UVEC[j].
    let mut jj = 0usize;
    while jj < BV {
        unsafe {
            PROD[tid] = k_t * s_scaled[jj];
        }
        thread::sync_threads();
        // Tree reduction: D_K=64 → 32 → 16 → 8 → 4 → 2 → 1
        let mut stride = 32usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PROD[tid] = PROD[tid] + PROD[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe { UVEC[jj] = PROD[0]; }
        }
        thread::sync_threads();
        jj += 1;
    }

    // ── Compute residual and S_out row (BV cols) in registers ──
    // r[j] = v[j] - u[j];   S_out[t, j] = S_scaled[t, j] + β * k_t * r[j]
    let mut s_out_row: [f32; BV] = [0.0; BV];
    let bk = beta_v * k_t;
    let mut jj2 = 0usize;
    while jj2 < BV {
        let v_j = v[bh * 64 + bv_col0 + jj2];
        let u_j = unsafe { UVEC[jj2] };
        let r_j = v_j - u_j;
        // S_out[t, jj2] = s_scaled[jj2] + (beta * k_t) * r_j
        let so = unsafe { core::intrinsics::fmuladdf32(bk, r_j, s_scaled[jj2]) };
        s_out_row[jj2] = so;
        jj2 += 1;
    }

    // ── Reduction 2: o[j] = sum_t q_t * s_out[t, j] ──
    let mut jj3 = 0usize;
    while jj3 < BV {
        unsafe {
            PROD[tid] = q_t * s_out_row[jj3];
        }
        thread::sync_threads();
        let mut stride = 32usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PROD[tid] = PROD[tid] + PROD[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe { OVEC[jj3] = PROD[0]; }
        }
        thread::sync_threads();
        jj3 += 1;
    }

    // ── Stores ──
    // Each thread writes its row of S_out (BV cols).
    let so_ptr = s_out.as_mut_ptr();
    let mut wj = 0usize;
    while wj < BV {
        unsafe {
            *so_ptr.add(s_row_base + bv_col0 + wj) = s_out_row[wj];
        }
        wj += 1;
    }
    // Threads 0..BV write the O vector.
    if tid < BV {
        let o_ptr = o.as_mut_ptr();
        unsafe {
            *o_ptr.add(bh * 64 + bv_col0 + tid) = OVEC[tid];
        }
    }
}

/// GDN decode kernel for d_k = 256 (Qwen3-Next decode shape).
/// Block = 256 threads, grid = (B*H, d_v / BV).  d_v=256, BV=32 → grid_y = 8.
#[kernel]
pub fn gdn_decode_dk256(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta: &[f32],
    s_in: &[f32],
    mut s_out: DisjointSlice<f32>,
    mut o: DisjointSlice<f32>,
    _d_k: u32,
    _d_v: u32,
) {
    static mut K_SH: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut Q_SH: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut PROD: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut UVEC: SharedArray<f32, 32> = SharedArray::UNINIT;
    static mut OVEC: SharedArray<f32, 32> = SharedArray::UNINIT;
    static mut AB: SharedArray<f32, 2> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let bh = thread::blockIdx_x() as usize;
    let bv_idx = thread::blockIdx_y() as usize;

    if tid == 0 {
        unsafe { AB[0] = alpha[bh]; }
    }
    if tid == 1 {
        unsafe { AB[1] = beta[bh]; }
    }

    let q_t = q[bh * 256 + tid];
    let k_t = k[bh * 256 + tid];
    unsafe {
        K_SH[tid] = k_t;
        Q_SH[tid] = q_t;
    }
    thread::sync_threads();
    let alpha_v = unsafe { AB[0] };
    let beta_v = unsafe { AB[1] };

    let s_row_base = (bh * 256 + tid) * 256; // d_v = 256
    let bv_col0 = bv_idx * BV;
    let mut s_scaled: [f32; BV] = [0.0; BV];
    let mut j = 0usize;
    while j < BV {
        let s = s_in[s_row_base + bv_col0 + j];
        s_scaled[j] = s * alpha_v;
        j += 1;
    }

    let mut jj = 0usize;
    while jj < BV {
        unsafe {
            PROD[tid] = k_t * s_scaled[jj];
        }
        thread::sync_threads();
        let mut stride = 128usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PROD[tid] = PROD[tid] + PROD[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe { UVEC[jj] = PROD[0]; }
        }
        thread::sync_threads();
        jj += 1;
    }

    let mut s_out_row: [f32; BV] = [0.0; BV];
    let bk = beta_v * k_t;
    let mut jj2 = 0usize;
    while jj2 < BV {
        let v_j = v[bh * 256 + bv_col0 + jj2];
        let u_j = unsafe { UVEC[jj2] };
        let r_j = v_j - u_j;
        let so = unsafe { core::intrinsics::fmuladdf32(bk, r_j, s_scaled[jj2]) };
        s_out_row[jj2] = so;
        jj2 += 1;
    }

    let mut jj3 = 0usize;
    while jj3 < BV {
        unsafe {
            PROD[tid] = q_t * s_out_row[jj3];
        }
        thread::sync_threads();
        let mut stride = 128usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PROD[tid] = PROD[tid] + PROD[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe { OVEC[jj3] = PROD[0]; }
        }
        thread::sync_threads();
        jj3 += 1;
    }

    let so_ptr = s_out.as_mut_ptr();
    let mut wj = 0usize;
    while wj < BV {
        unsafe {
            *so_ptr.add(s_row_base + bv_col0 + wj) = s_out_row[wj];
        }
        wj += 1;
    }
    if tid < BV {
        let o_ptr = o.as_mut_ptr();
        unsafe {
            *o_ptr.add(bh * 256 + bv_col0 + tid) = OVEC[tid];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Shape {
    name: &'static str,
    batch: usize,
    n_heads: usize,
    d_k: usize,
    d_v: usize,
}

const SHAPE_CORRECTNESS: Shape = Shape {
    name: "correctness",
    batch: 2,
    n_heads: 4,
    d_k: 64,
    d_v: 64,
};

const SHAPE_QWEN3_NEXT_DECODE: Shape = Shape {
    name: "qwen3_next_decode",
    batch: 1,
    n_heads: 16,
    d_k: 256,
    d_v: 256,
};

#[derive(Debug)]
struct Npy {
    shape: Vec<usize>,
    dtype: String,
    data: Vec<u8>,
}

impl Npy {
    fn num_elems(&self) -> usize {
        self.shape.iter().product()
    }
    fn as_f32(&self) -> Vec<f32> {
        let n = self.num_elems();
        let mut out = Vec::with_capacity(n);
        if self.dtype == "<f4" {
            for i in 0..n {
                let off = i * 4;
                let b = [
                    self.data[off],
                    self.data[off + 1],
                    self.data[off + 2],
                    self.data[off + 3],
                ];
                out.push(f32::from_le_bytes(b));
            }
        } else {
            panic!("as_f32 called on non-f4 npy: {}", self.dtype);
        }
        out
    }
}

fn load_npy(path: &str) -> Npy {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
    let mut magic = [0u8; 6];
    f.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"\x93NUMPY", "not a .npy: {}", path);
    let mut ver = [0u8; 2];
    f.read_exact(&mut ver).unwrap();
    let header_len = if ver[0] == 1 {
        let mut h = [0u8; 2];
        f.read_exact(&mut h).unwrap();
        u16::from_le_bytes(h) as usize
    } else {
        let mut h = [0u8; 4];
        f.read_exact(&mut h).unwrap();
        u32::from_le_bytes(h) as usize
    };
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header).unwrap();
    let header = String::from_utf8(header).unwrap();

    let dp = header
        .find("'descr':")
        .or_else(|| header.find("\"descr\":"))
        .unwrap();
    let sq1 = header[dp + 8..].find('\'').unwrap() + dp + 8;
    let sq2 = header[sq1 + 1..].find('\'').unwrap() + sq1 + 1;
    let dtype = header[sq1 + 1..sq2].to_string();

    let spos = header
        .find("'shape':")
        .or_else(|| header.find("\"shape\":"))
        .unwrap();
    let lp = header[spos..].find('(').unwrap() + spos;
    let rp = header[lp..].find(')').unwrap() + lp;
    let shape_str = &header[lp + 1..rp];
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();

    let num_elems: usize = shape.iter().product();
    let elem_size = if dtype == "<f2" {
        2
    } else if dtype == "<f4" {
        4
    } else {
        panic!("unsupported dtype {}", dtype)
    };
    let mut data = vec![0u8; num_elems * elem_size];
    f.read_exact(&mut data).unwrap();
    Npy { shape, dtype, data }
}

fn upload_f32(stream: &cuda_core::CudaStream, dst: &DeviceBuffer<f32>, src: &[f32]) {
    use cuda_core::IntoResult;
    let num_bytes = std::mem::size_of_val(src);
    assert!(num_bytes <= dst.num_bytes());
    stream.context().bind_to_thread().expect("bind ctx");
    unsafe {
        sys::cuMemcpyHtoDAsync_v2(
            dst.cu_deviceptr(),
            src.as_ptr() as *const _,
            num_bytes,
            stream.cu_stream(),
        )
        .result()
        .expect("htod");
    }
    stream.synchronize().unwrap();
}

fn run_shape(
    shape: &Shape,
    _ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &std::sync::Arc<cuda_core::CudaModule>,
    csv: &mut BufWriter<File>,
) {
    let inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    let q_path = format!("{}/gdn_{}_q_f32.npy", inputs_dir, shape.name);
    let k_path = format!("{}/gdn_{}_k_f32.npy", inputs_dir, shape.name);
    let v_path = format!("{}/gdn_{}_v_f32.npy", inputs_dir, shape.name);
    let alpha_path = format!("{}/gdn_{}_alpha_f32.npy", inputs_dir, shape.name);
    let beta_path = format!("{}/gdn_{}_beta_f32.npy", inputs_dir, shape.name);
    let s_in_path = format!("{}/gdn_{}_S_in_f32.npy", inputs_dir, shape.name);
    let s_out_path = format!("{}/gdn_{}_S_out_expected_f32.npy", inputs_dir, shape.name);

    if !Path::new(&q_path).exists() {
        panic!(
            "missing input {}. Run pytorch_reference_gdn.py to regenerate.",
            q_path
        );
    }

    let q_npy = load_npy(&q_path);
    let k_npy = load_npy(&k_path);
    let v_npy = load_npy(&v_path);
    let alpha_npy = load_npy(&alpha_path);
    let beta_npy = load_npy(&beta_path);
    let s_in_npy = load_npy(&s_in_path);
    let s_out_e_npy = load_npy(&s_out_path);
    // We compare o against the reference's f32 view of o_expected.
    // o_expected is stored as f16 in npy; we load alpha_f32/beta_f32 paths
    // which exist. For "o" we re-derive expected by reading the f16 path
    // and casting? Simpler: load _o_expected_f16.npy and convert to f32 here.
    let o_exp_f16_path = format!(
        "{}/gdn_{}_o_expected_f16.npy",
        inputs_dir, shape.name
    );
    let o_exp_npy = load_npy(&o_exp_f16_path);
    assert_eq!(o_exp_npy.dtype, "<f2");

    let q_host = q_npy.as_f32();
    let k_host = k_npy.as_f32();
    let v_host = v_npy.as_f32();
    let alpha_host = alpha_npy.as_f32();
    let beta_host = beta_npy.as_f32();
    let s_in_host = s_in_npy.as_f32();
    let s_out_e_host = s_out_e_npy.as_f32();

    // f16 → f32 conversion for o_expected.
    let o_exp_host: Vec<f32> = {
        let n = o_exp_npy.num_elems();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * 2;
            let bits = u16::from_le_bytes([o_exp_npy.data[off], o_exp_npy.data[off + 1]]);
            // Manual half→float conversion.
            let sign = (bits >> 15) & 0x1;
            let exp = (bits >> 10) & 0x1f;
            let mant = bits & 0x3ff;
            let f = if exp == 0 {
                if mant == 0 {
                    0.0_f32
                } else {
                    // subnormal
                    let m = mant as f32 / 1024.0;
                    m * 2f32.powi(-14)
                }
            } else if exp == 31 {
                if mant == 0 {
                    f32::INFINITY
                } else {
                    f32::NAN
                }
            } else {
                let m = 1.0_f32 + (mant as f32) / 1024.0;
                m * 2f32.powi(exp as i32 - 15)
            };
            out.push(if sign == 1 { -f } else { f });
        }
        out
    };

    let bh = shape.batch * shape.n_heads;
    let q_elems = bh * shape.d_k;
    let k_elems = bh * shape.d_k;
    let v_elems = bh * shape.d_v;
    let s_elems = bh * shape.d_k * shape.d_v;
    let o_elems = bh * shape.d_v;
    let ab_elems = bh;

    assert_eq!(q_host.len(), q_elems);
    assert_eq!(k_host.len(), k_elems);
    assert_eq!(v_host.len(), v_elems);
    assert_eq!(s_in_host.len(), s_elems);
    assert_eq!(s_out_e_host.len(), s_elems);
    assert_eq!(o_exp_host.len(), o_elems);
    assert_eq!(alpha_host.len(), ab_elems);
    assert_eq!(beta_host.len(), ab_elems);

    let q_dev = DeviceBuffer::from_host(stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(stream, &v_host).unwrap();
    let alpha_dev = DeviceBuffer::from_host(stream, &alpha_host).unwrap();
    let beta_dev = DeviceBuffer::from_host(stream, &beta_host).unwrap();
    let s_in_dev = DeviceBuffer::from_host(stream, &s_in_host).unwrap();
    let mut s_out_dev = DeviceBuffer::<f32>::zeroed(stream, s_elems).unwrap();
    let mut o_dev = DeviceBuffer::<f32>::zeroed(stream, o_elems).unwrap();

    upload_f32(stream, &q_dev, &q_host);
    upload_f32(stream, &k_dev, &k_host);
    upload_f32(stream, &v_dev, &v_host);
    upload_f32(stream, &alpha_dev, &alpha_host);
    upload_f32(stream, &beta_dev, &beta_host);
    upload_f32(stream, &s_in_dev, &s_in_host);

    println!(
        "[oxide-attn-gdn] shape={} (B={} H={} d_k={} d_v={}) state_MB={:.2}",
        shape.name,
        shape.batch,
        shape.n_heads,
        shape.d_k,
        shape.d_v,
        (s_elems * 4) as f64 / 1e6
    );

    let cfg = LaunchConfig {
        grid_dim: (bh as u32, (shape.d_v / BV) as u32, 1),
        block_dim: (shape.d_k as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let d_k_u = shape.d_k as u32;
    let d_v_u = shape.d_v as u32;

    // Rebind &mut for cuda_launch macro reborrow.
    let mut s_out_dev_mut = &mut s_out_dev;
    let mut o_dev_mut = &mut o_dev;

    {
        let s = stream.clone();
        let m = module.clone();
        if shape.d_k == 64 {
            cuda_launch! {
                kernel: gdn_decode_dk64,
                stream: s,
                module: m,
                config: cfg,
                args: [
                    slice(&q_dev),
                    slice(&k_dev),
                    slice(&v_dev),
                    slice(&alpha_dev),
                    slice(&beta_dev),
                    slice(&s_in_dev),
                    slice_mut(s_out_dev_mut),
                    slice_mut(o_dev_mut),
                    d_k_u,
                    d_v_u
                ]
            }
            .unwrap();
        } else {
            cuda_launch! {
                kernel: gdn_decode_dk256,
                stream: s,
                module: m,
                config: cfg,
                args: [
                    slice(&q_dev),
                    slice(&k_dev),
                    slice(&v_dev),
                    slice(&alpha_dev),
                    slice(&beta_dev),
                    slice(&s_in_dev),
                    slice_mut(s_out_dev_mut),
                    slice_mut(o_dev_mut),
                    d_k_u,
                    d_v_u
                ]
            }
            .unwrap();
        }
    }
    stream.synchronize().unwrap();

    // Re-borrow as immutable for read-back.
    let o_got = o_dev.to_host_vec(stream).unwrap();
    let s_out_got = s_out_dev.to_host_vec(stream).unwrap();

    // Correctness vs PyTorch reference.
    let mut max_abs_o: f32 = 0.0;
    for i in 0..o_elems {
        let d = (o_got[i] - o_exp_host[i]).abs();
        if d > max_abs_o {
            max_abs_o = d;
        }
    }
    let mut max_abs_s: f32 = 0.0;
    for i in 0..s_elems {
        let d = (s_out_got[i] - s_out_e_host[i]).abs();
        if d > max_abs_s {
            max_abs_s = d;
        }
    }

    // o_expected is f16, so the ground-truth has up to ~5e-4 quantization
    // noise from the f16 cast. f32 kernel result vs f32-of-f16-truth:
    // tolerance ≤ 1e-3 per ADR-0004 / wave-17 acceptance.
    let tol_o: f32 = 1e-3;
    let tol_s: f32 = 1e-3;
    let ok_o = max_abs_o <= tol_o;
    let ok_s = max_abs_s <= tol_s;

    println!(
        "[oxide-attn-gdn] {} correctness: max_abs(o)={:.3e} max_abs(S_out)={:.3e} (tol={}) -> {}",
        shape.name,
        max_abs_o,
        max_abs_s,
        tol_o,
        if ok_o && ok_s { "OK" } else { "FAIL" }
    );
    writeln!(
        csv,
        "oxide-attn-gdn,{},correctness,max_abs_o,{:.6e}",
        shape.name, max_abs_o
    )
    .unwrap();
    writeln!(
        csv,
        "oxide-attn-gdn,{},correctness,max_abs_S_out,{:.6e}",
        shape.name, max_abs_s
    )
    .unwrap();

    if !(ok_o && ok_s) {
        eprintln!(
            "WARN: shape {} failed correctness (max_abs_o={:.3e} max_abs_s={:.3e})",
            shape.name, max_abs_o, max_abs_s
        );
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_attn_gdn").expect("load module");

    let csv_path = "/home/codeseys/cuda-exploration/oxide-attn-gdn/results.csv";
    let csv_file = File::create(csv_path).expect("create csv");
    let mut csv = BufWriter::new(csv_file);
    writeln!(&mut csv, "impl,shape,kind,metric,value").unwrap();

    println!("[oxide-attn-gdn] Wave 17 W1d — cuda-oxide GDN decode (f32, no-TC)");
    println!("[oxide-attn-gdn] GPU: RTX 5090 sm_120, correctness-only run");

    run_shape(&SHAPE_CORRECTNESS, &ctx, &stream, &module, &mut csv);
    run_shape(&SHAPE_QWEN3_NEXT_DECODE, &ctx, &stream, &module, &mut csv);

    csv.flush().unwrap();
    println!("[oxide-attn-gdn] results.csv written");
}
