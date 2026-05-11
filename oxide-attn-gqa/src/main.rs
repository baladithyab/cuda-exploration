// Wave 16.1 — cuda-oxide GQA attention, f32 throughout, NO tensor cores.
//
// This is the "no-TC ceiling" data point: cuda-oxide v0.1.0 has no usable
// tensor-core API on consumer Blackwell sm_120 (Wave 14.4: zero `mma.sync`
// in source, wgmma is a placeholder, tcgen05 is sm_100a-only). So the
// cell is f32×f32 matmul with register microtile + custom softmax. Goal
// is to sit the cell next to the Wave 15 cuTile (165 TF) / cuBLAS (46 TF) /
// nvcc-WMMA (23 TF) numbers and see how slow cuda-oxide's no-TC path is.
//
// Three kernels:
//   gqa_qkt_kernel:   Q[b,nq,s,d] x K[b,nkv,s,d]^T  -> scores[b,nq,s,s] * 1/sqrt(d)
//   softmax_kernel:   row-wise softmax over last dim of scores
//   gqa_pv_kernel:    probs[b,nq,s,s] x V[b,nkv,s,d] -> out[b,nq,s,d]
//
// GQA broadcasting: Q head h_q reads K/V head h_kv = h_q / groups.
// The h_kv index is computed from blockIdx.z in the matmul kernels.
//
// Microtile pattern from oxide-matmul-tiled-microtile (Wave 7, 45 TF f32
// matmul): block 16x16 threads, each thread computes a 4x4 output tile ->
// 64x64 output per block. K-tile size BK=16, 16 FFMA per K-iter per thread.
// Adapted here for the two matmul shapes:
//   QKt:  (seq,d) x (d,seq) -> (seq,seq). d=128 -> 8 K-tiles (divides 16).
//   PV:   (seq,seq) x (seq,d) -> (seq,d). d=128, so grid x = 128/64 = 2.
//
// Softmax: one block per (b, h_q, row). Block = 128 threads. Each thread
// strides over the row and keeps a local max/sum; warp-shuffle + smem
// reduce. Scale (1/sqrt(d)) is fused into the max pass.
//
// Uses `core::intrinsics::fmuladdf32` in the inner matmul loop (Wave 3/7
// finding: libNVVM's FMA contractor misses runtime-bounded loops without
// this intrinsic; with it, PTX contains hardware FMA).

#![feature(core_intrinsics)]
#![allow(internal_features)]

use cuda_core::{CudaContext, CudaEvent, DeviceBuffer, LaunchConfig, sys};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};
use cuda_host::{cuda_launch, load_kernel_module};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

// ---------- kernels ----------

/// Q[b, h_q, row, k] x K[b, h_kv, col, k]^T = S[b, h_q, row, col] * scale
///
/// Output matrix (seq, seq). Each block covers 64x64 of the output.
/// 16x16 threads, each computing a 4x4 microtile.
/// K-tile size = 16 (along d_head). d_head = 128 gives 8 K-tiles.
///
/// Layout: Q is (B, n_q, S, d) row-major, K is (B, n_kv, S, d) row-major.
/// For a given (batch b, q-head h_q) pair the Q slice starts at
///   q_base = ((b * n_q) + h_q) * S * d
/// and the K slice (KV head h_kv = h_q / groups) at
///   k_base = ((b * n_kv) + h_q / groups) * S * d
/// Scores output at
///   s_base = ((b * n_q) + h_q) * S * S
///
/// The kernel is launched as grid=(S/64, S/64, B*n_q), block=(16,16,1).
/// gridZ index = b * n_q + h_q.
#[kernel]
pub fn gqa_qkt_kernel(
    q: &[f32],
    k: &[f32],
    mut scores: DisjointSlice<f32>,
    seq: u32,
    d_head: u32,
    n_q: u32,
    n_kv: u32,
    scale: f32,
) {
    static mut TILE_Q: SharedArray<f32, 1024> = SharedArray::UNINIT; // 64 x 16
    static mut TILE_K: SharedArray<f32, 1024> = SharedArray::UNINIT; // 16 x 64 (as loaded: transposed)

    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;
    let bx = thread::blockIdx_x() as usize;
    let by = thread::blockIdx_y() as usize;
    let bz = thread::blockIdx_z() as usize;

    let seq_us = seq as usize;
    let d_us = d_head as usize;
    let n_q_us = n_q as usize;
    let groups = (n_q / n_kv) as usize;

    // (batch, q-head) from bz
    let b = bz / n_q_us;
    let h_q = bz % n_q_us;
    let h_kv = h_q / groups;

    let q_base = ((b * n_q_us) + h_q) * seq_us * d_us;
    let k_base = ((b * (n_kv as usize)) + h_kv) * seq_us * d_us;
    let s_base = ((b * n_q_us) + h_q) * seq_us * seq_us;

    let row0 = by * 64 + ty * 4; // starting row in scores
    let col0 = bx * 64 + tx * 4; // starting col in scores
    let tid = ty * 16 + tx;
    let num_tiles = d_us / 16;

    // 4x4 scalar accumulators.
    let mut c00: f32 = 0.0; let mut c01: f32 = 0.0; let mut c02: f32 = 0.0; let mut c03: f32 = 0.0;
    let mut c10: f32 = 0.0; let mut c11: f32 = 0.0; let mut c12: f32 = 0.0; let mut c13: f32 = 0.0;
    let mut c20: f32 = 0.0; let mut c21: f32 = 0.0; let mut c22: f32 = 0.0; let mut c23: f32 = 0.0;
    let mut c30: f32 = 0.0; let mut c31: f32 = 0.0; let mut c32: f32 = 0.0; let mut c33: f32 = 0.0;

    let q_ptr = q.as_ptr();
    let k_ptr = k.as_ptr();

    let mut t: usize = 0;
    while t < num_tiles {
        let k_off = t * 16;

        // Cooperative load Q tile (64 rows x 16 cols of d_head) from Q[b,h_q,row,k].
        // 256 threads x 4 elems each = 1024 floats.
        let mut li: usize = 0;
        while li < 4 {
            let idx = tid + li * 256;
            let r = idx / 16;
            let kk = idx & 15;
            let gr = by * 64 + r;
            let gk = k_off + kk;
            unsafe {
                // Q[b,h_q,gr,gk]  -- row-major (S, d)
                let v = *q_ptr.add(q_base + gr * d_us + gk);
                TILE_Q[idx] = v;
            }
            li += 1;
        }

        // Cooperative load K tile for transpose: TILE_K[kk, cc] = K[b,h_kv,gc,gk]
        // We arrange TILE_K[kk*64 + cc] so inner loop pulls TILE_K[kk*64 + tx*4 + j].
        let mut lj: usize = 0;
        while lj < 4 {
            let idx = tid + lj * 256;
            let kk = idx / 64;
            let cc = idx & 63;
            let gk = k_off + kk;
            let gc = bx * 64 + cc;
            unsafe {
                // K is (S, d) per head; we want K[gc, gk]  (so transpose into TILE_K).
                let v = *k_ptr.add(k_base + gc * d_us + gk);
                TILE_K[idx] = v;
            }
            lj += 1;
        }
        thread::sync_threads();

        let ty4 = ty * 4;
        let tx4 = tx * 4;
        let mut kk: usize = 0;
        while kk < 16 {
            let a0: f32; let a1: f32; let a2: f32; let a3: f32;
            let b0: f32; let b1: f32; let b2: f32; let b3: f32;
            unsafe {
                a0 = TILE_Q[(ty4 + 0) * 16 + kk];
                a1 = TILE_Q[(ty4 + 1) * 16 + kk];
                a2 = TILE_Q[(ty4 + 2) * 16 + kk];
                a3 = TILE_Q[(ty4 + 3) * 16 + kk];
                b0 = TILE_K[kk * 64 + tx4 + 0];
                b1 = TILE_K[kk * 64 + tx4 + 1];
                b2 = TILE_K[kk * 64 + tx4 + 2];
                b3 = TILE_K[kk * 64 + tx4 + 3];
                c00 = core::intrinsics::fmuladdf32(a0, b0, c00);
                c01 = core::intrinsics::fmuladdf32(a0, b1, c01);
                c02 = core::intrinsics::fmuladdf32(a0, b2, c02);
                c03 = core::intrinsics::fmuladdf32(a0, b3, c03);
                c10 = core::intrinsics::fmuladdf32(a1, b0, c10);
                c11 = core::intrinsics::fmuladdf32(a1, b1, c11);
                c12 = core::intrinsics::fmuladdf32(a1, b2, c12);
                c13 = core::intrinsics::fmuladdf32(a1, b3, c13);
                c20 = core::intrinsics::fmuladdf32(a2, b0, c20);
                c21 = core::intrinsics::fmuladdf32(a2, b1, c21);
                c22 = core::intrinsics::fmuladdf32(a2, b2, c22);
                c23 = core::intrinsics::fmuladdf32(a2, b3, c23);
                c30 = core::intrinsics::fmuladdf32(a3, b0, c30);
                c31 = core::intrinsics::fmuladdf32(a3, b1, c31);
                c32 = core::intrinsics::fmuladdf32(a3, b2, c32);
                c33 = core::intrinsics::fmuladdf32(a3, b3, c33);
            }
            kk += 1;
        }
        thread::sync_threads();
        t += 1;
    }

    let s_ptr = scores.as_mut_ptr();
    unsafe {
        // Scale fused into the store.
        *s_ptr.add(s_base + (row0 + 0) * seq_us + col0 + 0) = c00 * scale;
        *s_ptr.add(s_base + (row0 + 0) * seq_us + col0 + 1) = c01 * scale;
        *s_ptr.add(s_base + (row0 + 0) * seq_us + col0 + 2) = c02 * scale;
        *s_ptr.add(s_base + (row0 + 0) * seq_us + col0 + 3) = c03 * scale;
        *s_ptr.add(s_base + (row0 + 1) * seq_us + col0 + 0) = c10 * scale;
        *s_ptr.add(s_base + (row0 + 1) * seq_us + col0 + 1) = c11 * scale;
        *s_ptr.add(s_base + (row0 + 1) * seq_us + col0 + 2) = c12 * scale;
        *s_ptr.add(s_base + (row0 + 1) * seq_us + col0 + 3) = c13 * scale;
        *s_ptr.add(s_base + (row0 + 2) * seq_us + col0 + 0) = c20 * scale;
        *s_ptr.add(s_base + (row0 + 2) * seq_us + col0 + 1) = c21 * scale;
        *s_ptr.add(s_base + (row0 + 2) * seq_us + col0 + 2) = c22 * scale;
        *s_ptr.add(s_base + (row0 + 2) * seq_us + col0 + 3) = c23 * scale;
        *s_ptr.add(s_base + (row0 + 3) * seq_us + col0 + 0) = c30 * scale;
        *s_ptr.add(s_base + (row0 + 3) * seq_us + col0 + 1) = c31 * scale;
        *s_ptr.add(s_base + (row0 + 3) * seq_us + col0 + 2) = c32 * scale;
        *s_ptr.add(s_base + (row0 + 3) * seq_us + col0 + 3) = c33 * scale;
    }
}

/// Row-wise softmax: scores[b, h_q, row, :] -> probs[b, h_q, row, :]
/// One block per (b*n_q*seq) row. Block = 128 threads = 4 warps.
/// Three passes: row max, exp+sum, divide. Grid-stride within the row.
#[kernel]
pub fn softmax_kernel(
    scores: &[f32],
    mut probs: DisjointSlice<f32>,
    seq: u32,
) {
    static mut WMAX: SharedArray<f32, 4> = SharedArray::UNINIT;
    static mut WSUM: SharedArray<f32, 4> = SharedArray::UNINIT;
    static mut ROW_MAX: SharedArray<f32, 1> = SharedArray::UNINIT;
    static mut ROW_SUM: SharedArray<f32, 1> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let bid = thread::blockIdx_x() as usize; // row index (flattened over b, h_q, s)
    let seq_us = seq as usize;
    let lane = warp::lane_id() as usize;
    let warp_id = tid >> 5;
    let bdim = thread::blockDim_x() as usize;

    let row_base = bid * seq_us;
    let s_ptr = scores.as_ptr();
    let p_ptr = probs.as_mut_ptr();

    // Pass 1: max
    let mut m: f32 = f32::NEG_INFINITY;
    let mut i = tid;
    while i < seq_us {
        let v = unsafe { *s_ptr.add(row_base + i) };
        if v > m { m = v; }
        i += bdim;
    }
    // warp reduce max
    let mut d = 16;
    while d > 0 {
        let other = warp::shuffle_xor_f32(m, d);
        if other > m { m = other; }
        d >>= 1;
    }
    if lane == 0 { unsafe { WMAX[warp_id] = m; } }
    thread::sync_threads();
    if warp_id == 0 {
        let mut v: f32 = if lane < 4 { unsafe { WMAX[lane] } } else { f32::NEG_INFINITY };
        let mut dd = 2;
        while dd > 0 {
            let other = warp::shuffle_xor_f32(v, dd);
            if other > v { v = other; }
            dd >>= 1;
        }
        if lane == 0 { unsafe { ROW_MAX[0] = v; } }
    }
    thread::sync_threads();
    let row_max = unsafe { ROW_MAX[0] };

    // Pass 2: exp(x - max), partial sum
    let mut s: f32 = 0.0;
    let mut j = tid;
    while j < seq_us {
        let v = unsafe { *s_ptr.add(row_base + j) };
        let e = unsafe { core::intrinsics::expf32(v - row_max) };
        unsafe { *p_ptr.add(row_base + j) = e; }
        s += e;
        j += bdim;
    }
    // warp reduce sum
    s += warp::shuffle_xor_f32(s, 16);
    s += warp::shuffle_xor_f32(s, 8);
    s += warp::shuffle_xor_f32(s, 4);
    s += warp::shuffle_xor_f32(s, 2);
    s += warp::shuffle_xor_f32(s, 1);
    if lane == 0 { unsafe { WSUM[warp_id] = s; } }
    thread::sync_threads();
    if warp_id == 0 {
        let mut v: f32 = if lane < 4 { unsafe { WSUM[lane] } } else { 0.0 };
        v += warp::shuffle_xor_f32(v, 2);
        v += warp::shuffle_xor_f32(v, 1);
        if lane == 0 { unsafe { ROW_SUM[0] = v; } }
    }
    thread::sync_threads();
    let row_sum = unsafe { ROW_SUM[0] };
    let inv = 1.0_f32 / row_sum;

    // Pass 3: normalize
    let mut l = tid;
    while l < seq_us {
        let e = unsafe { *p_ptr.add(row_base + l) };
        unsafe { *p_ptr.add(row_base + l) = e * inv; }
        l += bdim;
    }
}

/// probs[b,h_q,row,k] x V[b,h_kv,k,col] = out[b,h_q,row,col]
/// 4x4 microtile matmul.  probs shape (seq, seq), V (seq, d), out (seq, d).
/// Block covers 64 (row) x 64 (col) of output, but d_head=128 so gridX=2.
/// K-tile = 16; num_tiles = seq / 16.
#[kernel]
pub fn gqa_pv_kernel(
    probs: &[f32],
    v: &[f32],
    mut out: DisjointSlice<f32>,
    seq: u32,
    d_head: u32,
    n_q: u32,
    n_kv: u32,
) {
    static mut TILE_P: SharedArray<f32, 1024> = SharedArray::UNINIT; // 64 x 16
    static mut TILE_V: SharedArray<f32, 1024> = SharedArray::UNINIT; // 16 x 64

    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;
    let bx = thread::blockIdx_x() as usize;
    let by = thread::blockIdx_y() as usize;
    let bz = thread::blockIdx_z() as usize;

    let seq_us = seq as usize;
    let d_us = d_head as usize;
    let n_q_us = n_q as usize;
    let groups = (n_q / n_kv) as usize;

    let b = bz / n_q_us;
    let h_q = bz % n_q_us;
    let h_kv = h_q / groups;

    let p_base = ((b * n_q_us) + h_q) * seq_us * seq_us;
    let v_base = ((b * (n_kv as usize)) + h_kv) * seq_us * d_us;
    let o_base = ((b * n_q_us) + h_q) * seq_us * d_us;

    let row0 = by * 64 + ty * 4;
    let col0 = bx * 64 + tx * 4;
    let tid = ty * 16 + tx;
    let num_tiles = seq_us / 16;

    let mut c00: f32 = 0.0; let mut c01: f32 = 0.0; let mut c02: f32 = 0.0; let mut c03: f32 = 0.0;
    let mut c10: f32 = 0.0; let mut c11: f32 = 0.0; let mut c12: f32 = 0.0; let mut c13: f32 = 0.0;
    let mut c20: f32 = 0.0; let mut c21: f32 = 0.0; let mut c22: f32 = 0.0; let mut c23: f32 = 0.0;
    let mut c30: f32 = 0.0; let mut c31: f32 = 0.0; let mut c32: f32 = 0.0; let mut c33: f32 = 0.0;

    let p_ptr = probs.as_ptr();
    let v_ptr = v.as_ptr();

    let mut t: usize = 0;
    while t < num_tiles {
        let k_off = t * 16;

        // Load TILE_P[r, kk] = probs[row0_block+r, k_off+kk]  -- natural order
        let mut li: usize = 0;
        while li < 4 {
            let idx = tid + li * 256;
            let r = idx / 16;
            let kk = idx & 15;
            let gr = by * 64 + r;
            let gk = k_off + kk;
            unsafe {
                let vv = *p_ptr.add(p_base + gr * seq_us + gk);
                TILE_P[idx] = vv;
            }
            li += 1;
        }
        // Load TILE_V[kk, cc] = V[k_off+kk, col0_block+cc]
        let mut lj: usize = 0;
        while lj < 4 {
            let idx = tid + lj * 256;
            let kk = idx / 64;
            let cc = idx & 63;
            let gk = k_off + kk;
            let gc = bx * 64 + cc;
            unsafe {
                let vv = *v_ptr.add(v_base + gk * d_us + gc);
                TILE_V[idx] = vv;
            }
            lj += 1;
        }
        thread::sync_threads();

        let ty4 = ty * 4;
        let tx4 = tx * 4;
        let mut kk: usize = 0;
        while kk < 16 {
            let a0: f32; let a1: f32; let a2: f32; let a3: f32;
            let b0: f32; let b1: f32; let b2: f32; let b3: f32;
            unsafe {
                a0 = TILE_P[(ty4 + 0) * 16 + kk];
                a1 = TILE_P[(ty4 + 1) * 16 + kk];
                a2 = TILE_P[(ty4 + 2) * 16 + kk];
                a3 = TILE_P[(ty4 + 3) * 16 + kk];
                b0 = TILE_V[kk * 64 + tx4 + 0];
                b1 = TILE_V[kk * 64 + tx4 + 1];
                b2 = TILE_V[kk * 64 + tx4 + 2];
                b3 = TILE_V[kk * 64 + tx4 + 3];
                c00 = core::intrinsics::fmuladdf32(a0, b0, c00);
                c01 = core::intrinsics::fmuladdf32(a0, b1, c01);
                c02 = core::intrinsics::fmuladdf32(a0, b2, c02);
                c03 = core::intrinsics::fmuladdf32(a0, b3, c03);
                c10 = core::intrinsics::fmuladdf32(a1, b0, c10);
                c11 = core::intrinsics::fmuladdf32(a1, b1, c11);
                c12 = core::intrinsics::fmuladdf32(a1, b2, c12);
                c13 = core::intrinsics::fmuladdf32(a1, b3, c13);
                c20 = core::intrinsics::fmuladdf32(a2, b0, c20);
                c21 = core::intrinsics::fmuladdf32(a2, b1, c21);
                c22 = core::intrinsics::fmuladdf32(a2, b2, c22);
                c23 = core::intrinsics::fmuladdf32(a2, b3, c23);
                c30 = core::intrinsics::fmuladdf32(a3, b0, c30);
                c31 = core::intrinsics::fmuladdf32(a3, b1, c31);
                c32 = core::intrinsics::fmuladdf32(a3, b2, c32);
                c33 = core::intrinsics::fmuladdf32(a3, b3, c33);
            }
            kk += 1;
        }
        thread::sync_threads();
        t += 1;
    }

    let o_ptr = out.as_mut_ptr();
    unsafe {
        *o_ptr.add(o_base + (row0 + 0) * d_us + col0 + 0) = c00;
        *o_ptr.add(o_base + (row0 + 0) * d_us + col0 + 1) = c01;
        *o_ptr.add(o_base + (row0 + 0) * d_us + col0 + 2) = c02;
        *o_ptr.add(o_base + (row0 + 0) * d_us + col0 + 3) = c03;
        *o_ptr.add(o_base + (row0 + 1) * d_us + col0 + 0) = c10;
        *o_ptr.add(o_base + (row0 + 1) * d_us + col0 + 1) = c11;
        *o_ptr.add(o_base + (row0 + 1) * d_us + col0 + 2) = c12;
        *o_ptr.add(o_base + (row0 + 1) * d_us + col0 + 3) = c13;
        *o_ptr.add(o_base + (row0 + 2) * d_us + col0 + 0) = c20;
        *o_ptr.add(o_base + (row0 + 2) * d_us + col0 + 1) = c21;
        *o_ptr.add(o_base + (row0 + 2) * d_us + col0 + 2) = c22;
        *o_ptr.add(o_base + (row0 + 2) * d_us + col0 + 3) = c23;
        *o_ptr.add(o_base + (row0 + 3) * d_us + col0 + 0) = c30;
        *o_ptr.add(o_base + (row0 + 3) * d_us + col0 + 1) = c31;
        *o_ptr.add(o_base + (row0 + 3) * d_us + col0 + 2) = c32;
        *o_ptr.add(o_base + (row0 + 3) * d_us + col0 + 3) = c33;
    }
}

// ---------- host code ----------

#[derive(Debug, Clone)]
struct Shape {
    name: &'static str,
    batch: usize,
    seq: usize,
    n_q: usize,
    n_kv: usize,
    d_head: usize,
}

const SHAPE_CORRECTNESS: Shape = Shape {
    name: "correctness",
    batch: 1,
    seq: 128,
    n_q: 4,
    n_kv: 2,
    d_head: 64,
};
const SHAPE_BENCH: Shape = Shape {
    name: "llama3_8b",
    batch: 1,
    seq: 2048,
    n_q: 32,
    n_kv: 8,
    d_head: 128,
};

#[derive(Debug)]
struct Npy {
    shape: Vec<usize>,
    dtype: String, // "<f2" or "<f4"
    data: Vec<u8>,
}

impl Npy {
    fn elem_size(&self) -> usize {
        if self.dtype == "<f2" { 2 } else { 4 }
    }
    fn num_elems(&self) -> usize {
        self.shape.iter().product()
    }
    fn as_f32(&self) -> Vec<f32> {
        let n = self.num_elems();
        let mut out = Vec::with_capacity(n);
        if self.dtype == "<f4" {
            for i in 0..n {
                let off = i * 4;
                let b = [self.data[off], self.data[off + 1], self.data[off + 2], self.data[off + 3]];
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

    // descr
    let dp = header.find("'descr':").or_else(|| header.find("\"descr\":")).unwrap();
    let sq1 = header[dp + 8..].find('\'').unwrap() + dp + 8;
    let sq2 = header[sq1 + 1..].find('\'').unwrap() + sq1 + 1;
    let dtype = header[sq1 + 1..sq2].to_string();

    // shape
    let spos = header.find("'shape':").or_else(|| header.find("\"shape\":")).unwrap();
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
    let elem_size = if dtype == "<f2" { 2 } else if dtype == "<f4" { 4 } else {
        panic!("unsupported dtype {}", dtype)
    };
    let mut data = vec![0u8; num_elems * elem_size];
    f.read_exact(&mut data).unwrap();

    Npy { shape, dtype, data }
}

fn save_npy_f32(path: &str, data: &[f32], shape: &[usize]) {
    let mut f = File::create(path).unwrap();
    f.write_all(b"\x93NUMPY").unwrap();
    f.write_all(&[1u8, 0u8]).unwrap();
    let shape_str = shape.iter().map(|d| format!("{},", d)).collect::<String>();
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}",
        shape_str
    );
    // pad to 16-byte alignment
    let total = 10 + header.len() + 1; // +1 for newline
    let pad = (16 - (total % 16)) % 16;
    let mut header_padded = header;
    for _ in 0..pad { header_padded.push(' '); }
    header_padded.push('\n');
    let hl = header_padded.len() as u16;
    f.write_all(&hl.to_le_bytes()).unwrap();
    f.write_all(header_padded.as_bytes()).unwrap();
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    f.write_all(bytes).unwrap();
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

fn run_attention(
    shape: &Shape,
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &std::sync::Arc<cuda_core::CudaModule>,
    q_dev: &DeviceBuffer<f32>,
    k_dev: &DeviceBuffer<f32>,
    v_dev: &DeviceBuffer<f32>,
    scores_dev: &mut DeviceBuffer<f32>,
    probs_dev: &mut DeviceBuffer<f32>,
    out_dev: &mut DeviceBuffer<f32>,
    time_stages: bool,
) -> (f64, f64, f64, f64) {
    // returns (total_ms, qkt_ms, sm_ms, pv_ms)
    let seq = shape.seq as u32;
    let d = shape.d_head as u32;
    let n_q = shape.n_q as u32;
    let n_kv = shape.n_kv as u32;
    let scale = 1.0_f32 / (shape.d_head as f32).sqrt();

    // Rebind &mut params so cuda_launch! macro can reborrow across iterations.
    let mut scores_dev = scores_dev;
    let mut probs_dev = probs_dev;
    let mut out_dev = out_dev;

    let b_nq = shape.batch * shape.n_q;

    let cfg_qkt = LaunchConfig {
        grid_dim: ((shape.seq / 64) as u32, (shape.seq / 64) as u32, b_nq as u32),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    let cfg_softmax = LaunchConfig {
        grid_dim: ((shape.batch * shape.n_q * shape.seq) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg_pv = LaunchConfig {
        grid_dim: ((shape.d_head / 64) as u32, (shape.seq / 64) as u32, b_nq as u32),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    if time_stages {
        let ev_a = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
        let ev_b = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
        let ev_c = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
        let ev_d = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();

        ev_a.record(stream).unwrap();
        {
            let s = stream.clone(); let m = module.clone();
            cuda_launch! {
                kernel: gqa_qkt_kernel, stream: s, module: m, config: cfg_qkt,
                args: [slice(q_dev), slice(k_dev), slice_mut(scores_dev), seq, d, n_q, n_kv, scale]
            }.unwrap();
        }
        ev_b.record(stream).unwrap();
        {
            let s = stream.clone(); let m = module.clone();
            cuda_launch! {
                kernel: softmax_kernel, stream: s, module: m, config: cfg_softmax,
                args: [slice(scores_dev), slice_mut(probs_dev), seq]
            }.unwrap();
        }
        ev_c.record(stream).unwrap();
        {
            let s = stream.clone(); let m = module.clone();
            cuda_launch! {
                kernel: gqa_pv_kernel, stream: s, module: m, config: cfg_pv,
                args: [slice(probs_dev), slice(v_dev), slice_mut(out_dev), seq, d, n_q, n_kv]
            }.unwrap();
        }
        ev_d.record(stream).unwrap();
        stream.synchronize().unwrap();
        let qkt = ev_a.elapsed_ms(&ev_b).unwrap() as f64;
        let sm = ev_b.elapsed_ms(&ev_c).unwrap() as f64;
        let pv = ev_c.elapsed_ms(&ev_d).unwrap() as f64;
        let tot = ev_a.elapsed_ms(&ev_d).unwrap() as f64;
        (tot, qkt, sm, pv)
    } else {
        let ev_a = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
        let ev_d = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
        ev_a.record(stream).unwrap();
        {
            let s = stream.clone(); let m = module.clone();
            cuda_launch! {
                kernel: gqa_qkt_kernel, stream: s, module: m, config: cfg_qkt,
                args: [slice(q_dev), slice(k_dev), slice_mut(scores_dev), seq, d, n_q, n_kv, scale]
            }.unwrap();
        }
        {
            let s = stream.clone(); let m = module.clone();
            cuda_launch! {
                kernel: softmax_kernel, stream: s, module: m, config: cfg_softmax,
                args: [slice(scores_dev), slice_mut(probs_dev), seq]
            }.unwrap();
        }
        {
            let s = stream.clone(); let m = module.clone();
            cuda_launch! {
                kernel: gqa_pv_kernel, stream: s, module: m, config: cfg_pv,
                args: [slice(probs_dev), slice(v_dev), slice_mut(out_dev), seq, d, n_q, n_kv]
            }.unwrap();
        }
        ev_d.record(stream).unwrap();
        stream.synchronize().unwrap();
        let tot = ev_a.elapsed_ms(&ev_d).unwrap() as f64;
        (tot, 0.0, 0.0, 0.0)
    }
}

fn gqa_flops(s: &Shape) -> f64 {
    // 4 * batch * n_q * seq^2 * d_head
    4.0 * (s.batch as f64) * (s.n_q as f64) * (s.seq as f64) * (s.seq as f64) * (s.d_head as f64)
}

fn run_shape(
    shape: &Shape,
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &std::sync::Arc<cuda_core::CudaModule>,
    csv: &mut BufWriter<File>,
    iters: usize,
    time_stages: bool,
) {
    let inputs_dir = "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    let q_path = format!("{}/gqa_{}_q_f32.npy", inputs_dir, shape.name);
    let k_path = format!("{}/gqa_{}_k_f32.npy", inputs_dir, shape.name);
    let v_path = format!("{}/gqa_{}_v_f32.npy", inputs_dir, shape.name);
    let e_path = format!("{}/gqa_{}_expected_f32.npy", inputs_dir, shape.name);

    if !Path::new(&q_path).exists() {
        panic!("missing input {}. Run pytorch_reference.py to regenerate.", q_path);
    }

    let q_npy = load_npy(&q_path);
    let k_npy = load_npy(&k_path);
    let v_npy = load_npy(&v_path);
    let e_npy = load_npy(&e_path);
    assert_eq!(q_npy.dtype, "<f4");
    assert_eq!(k_npy.dtype, "<f4");
    assert_eq!(v_npy.dtype, "<f4");

    let q_host = q_npy.as_f32();
    let k_host = k_npy.as_f32();
    let v_host = v_npy.as_f32();
    let e_host = e_npy.as_f32();

    let q_elems = shape.batch * shape.n_q * shape.seq * shape.d_head;
    let kv_elems = shape.batch * shape.n_kv * shape.seq * shape.d_head;
    let scores_elems = shape.batch * shape.n_q * shape.seq * shape.seq;
    let out_elems = q_elems;

    assert_eq!(q_host.len(), q_elems);
    assert_eq!(k_host.len(), kv_elems);
    assert_eq!(v_host.len(), kv_elems);
    assert_eq!(e_host.len(), out_elems);

    let q_dev = DeviceBuffer::from_host(stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(stream, &v_host).unwrap();
    let mut scores_dev = DeviceBuffer::<f32>::zeroed(stream, scores_elems).unwrap();
    let mut probs_dev = DeviceBuffer::<f32>::zeroed(stream, scores_elems).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(stream, out_elems).unwrap();

    upload_f32(stream, &q_dev, &q_host);
    upload_f32(stream, &k_dev, &k_host);
    upload_f32(stream, &v_dev, &v_host);

    println!("[oxide-attn-gqa] shape={} (B={} S={} n_q={} n_kv={} d={}) scores_MB={:.1}",
        shape.name, shape.batch, shape.seq, shape.n_q, shape.n_kv, shape.d_head,
        (scores_elems * 4) as f64 / 1e6);

    // warmup
    let _ = run_attention(shape, ctx, stream, module, &q_dev, &k_dev, &v_dev,
        &mut scores_dev, &mut probs_dev, &mut out_dev, false);

    // correctness check
    let got = out_dev.to_host_vec(stream).unwrap();
    let mut max_abs: f32 = 0.0;
    let mut max_rel: f32 = 0.0;
    for i in 0..out_elems {
        let d = (got[i] - e_host[i]).abs();
        if d > max_abs { max_abs = d; }
        let e_abs = e_host[i].abs().max(1e-6);
        let r = d / e_abs;
        if r > max_rel { max_rel = r; }
    }
    let atol = 1e-5_f32; // f32 tolerance
    let rtol = 1e-5_f32;
    let loose_atol = 1e-3_f32; // naive f32 3-pass attention: softmax denorm drifts
    let loose_rtol = 1e-3_f32;
    let ok_tight = max_abs < atol || max_rel < rtol;
    let ok_loose = max_abs < loose_atol && max_rel < loose_rtol;
    println!(
        "[oxide-attn-gqa] {} correctness: max_abs={:.3e} max_rel={:.3e} (tight atol={} -> {}; loose atol={} -> {})",
        shape.name, max_abs, max_rel, atol, if ok_tight {"OK"} else {"miss"},
        loose_atol, if ok_loose {"OK"} else {"FAIL"}
    );
    // Save our output for inspection (small correctness only).
    if shape.name == "correctness" {
        let p = format!("/home/codeseys/cuda-exploration/oxide-attn-gqa/out_{}.npy", shape.name);
        save_npy_f32(&p, &got, &[shape.batch, shape.n_q, shape.seq, shape.d_head]);
    }

    // timed iters
    let total_flops = gqa_flops(shape);
    let mut ms_list: Vec<f64> = Vec::with_capacity(iters);
    let mut qkt_list: Vec<f64> = Vec::with_capacity(iters);
    let mut sm_list: Vec<f64> = Vec::with_capacity(iters);
    let mut pv_list: Vec<f64> = Vec::with_capacity(iters);
    for i in 0..iters {
        let t0 = Instant::now();
        let (tot, qkt, sm, pv) = run_attention(shape, ctx, stream, module,
            &q_dev, &k_dev, &v_dev,
            &mut scores_dev, &mut probs_dev, &mut out_dev, time_stages);
        let _cpu = t0.elapsed().as_secs_f64() * 1000.0;
        let tflops = total_flops / (tot * 1e-3) / 1e12;
        if time_stages {
            println!("[oxide-attn-gqa] {} iter={} total={:.3}ms qkt={:.3} sm={:.3} pv={:.3} tflops={:.2}",
                shape.name, i, tot, qkt, sm, pv, tflops);
            writeln!(csv, "oxide-attn-gqa,{},{},{:.6},{:.6},{:.6},{:.6},{:.3}",
                shape.name, i, tot, qkt, sm, pv, tflops).unwrap();
        } else {
            println!("[oxide-attn-gqa] {} iter={} total={:.3}ms tflops={:.2}",
                shape.name, i, tot, tflops);
            writeln!(csv, "oxide-attn-gqa,{},{},{:.6},,,,{:.3}",
                shape.name, i, tot, tflops).unwrap();
        }
        ms_list.push(tot); qkt_list.push(qkt); sm_list.push(sm); pv_list.push(pv);
    }
    let mut sorted = ms_list.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = sorted[0];
    let med = sorted[sorted.len()/2];
    let best_tf = total_flops / (best * 1e-3) / 1e12;
    let med_tf = total_flops / (med * 1e-3) / 1e12;
    println!("[oxide-attn-gqa] {} SUMMARY best={:.3}ms ({:.2} TF) median={:.3}ms ({:.2} TF)",
        shape.name, best, best_tf, med, med_tf);
    if time_stages {
        let avg = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        let q_avg = avg(&qkt_list); let s_avg = avg(&sm_list); let p_avg = avg(&pv_list);
        let t_avg = avg(&ms_list);
        println!("[oxide-attn-gqa] {} per-stage avg: qkt={:.3}ms ({:.1}%) sm={:.3}ms ({:.1}%) pv={:.3}ms ({:.1}%) total={:.3}ms",
            shape.name, q_avg, q_avg/t_avg*100.0, s_avg, s_avg/t_avg*100.0, p_avg, p_avg/t_avg*100.0, t_avg);
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_attn_gqa").expect("load module");

    let csv_path = "/home/codeseys/cuda-exploration/oxide-attn-gqa/results.csv";
    let csv_file = File::create(csv_path).expect("create csv");
    let mut csv = BufWriter::new(csv_file);
    writeln!(&mut csv, "impl,shape,iter,total_ms,qkt_ms,softmax_ms,pv_ms,TFLOPS").unwrap();

    println!("[oxide-attn-gqa] Wave 16.1 — cuda-oxide GQA (f32, no-TC)");
    println!("[oxide-attn-gqa] GPU: RTX 5090 sm_120, 1 warmup + 10 timed iters");

    // Correctness first.
    run_shape(&SHAPE_CORRECTNESS, &ctx, &stream, &module, &mut csv, 3, false);

    // Bench shape with per-stage timing.
    run_shape(&SHAPE_BENCH, &ctx, &stream, &module, &mut csv, 10, true);

    csv.flush().unwrap();
    println!("[oxide-attn-gqa] results.csv written");
}
