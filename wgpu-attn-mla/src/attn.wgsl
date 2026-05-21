// wgpu-attn-mla — naive 3-pass Multi-Head Latent Attention in WGSL.
//
// Wave C1.6: WebGPU port of MLA. Closes the WebGPU column for MLA in the
// Rosetta Stone matrix. Algorithmic reference: cuda-attn-mla/attn_mla.cu.
//
// Structural difference vs GQA:
//   * No GQA broadcast: Q, K, V all share the same head index h ∈ [0, n_h).
//     i.e. groups = 1 effectively. Q indexing therefore equals K/V indexing
//     on h. (This makes MLA addressing slightly *simpler* than GQA — there's
//     no integer-divide groups mapping.)
//   * Non-square per-head dims: qk_head_dim (= 96 corr / 192 bench) is the
//     contraction dim of QK^T, while d_v (= 64 corr / 128 bench) is the
//     contraction dim's *other side* of the PV product. So Q,K are shape
//     [B, n_h, S, qk] and V is [B, n_h, S, d_v]. Caller passes both QK and
//     Dv as separate uniforms.
//
// Layout:
//   Q : (B, n_h, S, qk) f32
//   K : (B, n_h, S, qk) f32
//   V : (B, n_h, S, d_v) f32
//   scores : (B, n_h, S, S) f32   intermediate, doubles as probs after softmax
//   out    : (B, n_h, S, d_v) f32
//
// f32 throughout. WGSL has an `f16` feature (wgpu::Features::SHADER_F16) but
// it's unavailable on llvmpipe/WSL2 and many backends. The reference *.npy
// inputs are loaded as f32 (mla_*_f32.npy), matching all other wave15/17
// reference cells. WebGPU has NO tensor cores, so this is FFMA-only — that
// is the headline expectation.
//
// FLOPs per forward (matches cuda-attn-mla/analysis/.../flops_mla.py):
//   2 · B · n_h · S² · qk        (QK^T)
//   + 2 · B · n_h · S² · d_v     (PV)
//   = 2 · B · n_h · S² · (qk + d_v)
// Softmax FLOPs ignored for the headline (small, O(S^2) not O(S^2 · D)).

struct Params {
    batch : u32,
    n_h   : u32,    // number of heads (Q,K,V share)
    seq   : u32,
    qk    : u32,    // qk_head_dim (contraction dim for QK^T)
    d_v   : u32,    // d_v (contraction "other side" for PV)
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
    scale : f32,    // 1 / sqrt(qk)
    _pad3 : f32,
    _pad4 : f32,
    _pad5 : f32,
};

@group(0) @binding(0) var<uniform>              P      : Params;
@group(0) @binding(1) var<storage, read>        Q      : array<f32>;
@group(0) @binding(2) var<storage, read>        K      : array<f32>;
@group(0) @binding(3) var<storage, read>        V      : array<f32>;
@group(0) @binding(4) var<storage, read_write>  Scores : array<f32>; // doubles as probs
@group(0) @binding(5) var<storage, read_write>  Out    : array<f32>;

// MLA: same head index for Q,K,V — no GQA divisor.
fn qk_idx(b: u32, h: u32, s: u32, d: u32) -> u32 {
    return ((b * P.n_h + h) * P.seq + s) * P.qk + d;
}
fn v_idx(b: u32, h: u32, s: u32, d: u32) -> u32 {
    return ((b * P.n_h + h) * P.seq + s) * P.d_v + d;
}
fn s_idx(b: u32, h: u32, i: u32, j: u32) -> u32 {
    return ((b * P.n_h + h) * P.seq + i) * P.seq + j;
}

// =============================================================
// Pass 1: QK^T with scale.
//   scores[b,h,i,j] = scale * sum_k Q[b,h,i,k] * K[b,h,j,k]
//   gid.x = j  gid.y = i  gid.z = b*n_h + h
// =============================================================
@compute @workgroup_size(16, 16, 1)
fn mla_qkt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j   = gid.x;
    let i   = gid.y;
    let bh  = gid.z;
    if (i >= P.seq || j >= P.seq) { return; }
    let total_bh = P.batch * P.n_h;
    if (bh >= total_bh) { return; }

    let b = bh / P.n_h;
    let h = bh % P.n_h;

    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < P.qk; k = k + 1u) {
        acc = acc + Q[qk_idx(b, h, i, k)] * K[qk_idx(b, h, j, k)];
    }
    Scores[s_idx(b, h, i, j)] = acc * P.scale;
}

// =============================================================
// Pass 2: row-wise numerically-stable softmax (in place on Scores).
//   One workgroup per row (b, h, i); WG_SOFTMAX threads cooperate over j.
// =============================================================
const WG_SOFTMAX : u32 = 256u;

var<workgroup> wg_scratch : array<f32, WG_SOFTMAX>;

@compute @workgroup_size(256, 1, 1)
fn softmax(
    @builtin(workgroup_id)        wg_id : vec3<u32>,
    @builtin(local_invocation_id) lid   : vec3<u32>,
) {
    let bhi = wg_id.x;
    let total = P.batch * P.n_h * P.seq;
    if (bhi >= total) { return; }

    let seq = P.seq;
    let i_in_h = bhi % seq;
    let bh     = bhi / seq;
    let h      = bh  % P.n_h;
    let b      = bh  / P.n_h;

    let row_base = s_idx(b, h, i_in_h, 0u);

    let tid = lid.x;
    var m : f32 = -3.4028235e38;
    var j : u32 = tid;
    loop {
        if (j >= seq) { break; }
        let v = Scores[row_base + j];
        if (v > m) { m = v; }
        j = j + WG_SOFTMAX;
    }
    wg_scratch[tid] = m;
    workgroupBarrier();
    var stride : u32 = WG_SOFTMAX / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            let other = wg_scratch[tid + stride];
            if (other > wg_scratch[tid]) { wg_scratch[tid] = other; }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max : f32 = wg_scratch[0];
    workgroupBarrier();

    var s : f32 = 0.0;
    j = tid;
    loop {
        if (j >= seq) { break; }
        let e = exp(Scores[row_base + j] - row_max);
        Scores[row_base + j] = e;
        s = s + e;
        j = j + WG_SOFTMAX;
    }
    wg_scratch[tid] = s;
    workgroupBarrier();
    stride = WG_SOFTMAX / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            wg_scratch[tid] = wg_scratch[tid] + wg_scratch[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum : f32 = wg_scratch[0];
    workgroupBarrier();

    let inv : f32 = 1.0 / row_sum;
    j = tid;
    loop {
        if (j >= seq) { break; }
        Scores[row_base + j] = Scores[row_base + j] * inv;
        j = j + WG_SOFTMAX;
    }
}

// =============================================================
// Pass 3: P @ V (single head, no GQA broadcast).
//   out[b,h,i,d] = sum_j probs[b,h,i,j] * V[b,h,j,d]
//   gid.x = d (0..d_v)  gid.y = i  gid.z = b*n_h + h
// =============================================================
@compute @workgroup_size(16, 16, 1)
fn mla_pv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d   = gid.x;
    let i   = gid.y;
    let bh  = gid.z;
    if (i >= P.seq || d >= P.d_v) { return; }
    let total_bh = P.batch * P.n_h;
    if (bh >= total_bh) { return; }

    let b = bh / P.n_h;
    let h = bh % P.n_h;

    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < P.seq; j = j + 1u) {
        acc = acc + Scores[s_idx(b, h, i, j)] * V[v_idx(b, h, j, d)];
    }
    Out[v_idx(b, h, i, d)] = acc;
}
