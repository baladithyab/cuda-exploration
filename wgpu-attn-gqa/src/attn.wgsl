// wgpu-attn-gqa — naive 3-pass Grouped-Query Attention in WGSL.
//
// Layout (matches the PyTorch reference / all other wave15 cells):
//   Q : (B, Nq,  S, D) f32, row-major
//   K : (B, Nkv, S, D) f32, row-major
//   V : (B, Nkv, S, D) f32, row-major
//   scores : (B, Nq, S, S) f32   -- intermediate from gqa_qkt
//   probs  : (B, Nq, S, S) f32   -- intermediate from softmax (in place on scores)
//   out    : (B, Nq, S, D) f32
//
// GQA broadcasting: for query head h_q in [0, Nq),
//   h_kv = h_q / groups        where groups = Nq / Nkv
// K and V are read from h_kv; each KV head is shared by `groups` query heads.
//
// f32 throughout. WGSL does have an `f16` feature (wgpu::Features::SHADER_F16),
// but it is unavailable on most llvmpipe / WSL paths and on many mobile
// backends. For a portable correctness artifact, f32 is the safe choice —
// and it also matches the *_f32.npy inputs of the shared wave15 reference.
//
// Three entry points (one compute pipeline per pass):
//
//   gqa_qkt  (B*Nq, S, S) -- workgroup_size(16,16,1)
//     scores[b, h_q, i, j] = sum_d Q[b, h_q, i, d] * K[b, h_kv, j, d] * scale
//
//   softmax  (B*Nq*S)   -- workgroup_size(256,1,1), one workgroup per row
//     probs[b, h_q, i, :] = softmax(scores[b, h_q, i, :])
//     numerically-stable: subtract rowmax, exp, divide by rowsum.
//
//   gqa_pv   (B*Nq, S, D) -- workgroup_size(16,16,1)
//     out[b, h_q, i, d] = sum_j probs[b, h_q, i, j] * V[b, h_kv, j, d]
//
// Naga translates this to SPIR-V for Vulkan and HLSL/DXIL for DX12. The
// generated SPIR-V is then compiled by the backend driver. WGSL's robust-access
// semantics (OOB reads return zero, OOB writes dropped) mean the explicit
// `if (i >= ...) return;` guards are belt-and-suspenders — they also help the
// downstream compiler elide tail iterations.

struct Params {
    batch : u32,
    n_q   : u32,
    n_kv  : u32,
    seq   : u32,
    d_head: u32,
    groups: u32,   // n_q / n_kv
    _pad0 : u32,
    _pad1 : u32,
    scale : f32,   // 1 / sqrt(d_head)
    _pad2 : f32,
    _pad3 : f32,
    _pad4 : f32,
};

@group(0) @binding(0) var<uniform>              P      : Params;
@group(0) @binding(1) var<storage, read>        Q      : array<f32>;
@group(0) @binding(2) var<storage, read>        K      : array<f32>;
@group(0) @binding(3) var<storage, read>        V      : array<f32>;
@group(0) @binding(4) var<storage, read_write>  Scores : array<f32>; // doubles as probs in pass 2
@group(0) @binding(5) var<storage, read_write>  Out    : array<f32>;

fn q_idx(b: u32, h_q: u32, s: u32, d: u32) -> u32 {
    return ((b * P.n_q + h_q) * P.seq + s) * P.d_head + d;
}
fn kv_idx(b: u32, h_kv: u32, s: u32, d: u32) -> u32 {
    return ((b * P.n_kv + h_kv) * P.seq + s) * P.d_head + d;
}
fn s_idx(b: u32, h_q: u32, i: u32, j: u32) -> u32 {
    return ((b * P.n_q + h_q) * P.seq + i) * P.seq + j;
}

// =============================================================
// Pass 1: QK^T with GQA broadcasting + scale.
//   gid.x = j   (key position,   0..seq)
//   gid.y = i   (query position, 0..seq)
//   gid.z = b*n_q + h_q
// =============================================================
@compute @workgroup_size(16, 16, 1)
fn gqa_qkt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j   = gid.x;
    let i   = gid.y;
    let bhq = gid.z;
    if (i >= P.seq || j >= P.seq) { return; }
    let total_bhq = P.batch * P.n_q;
    if (bhq >= total_bhq) { return; }

    let b   = bhq / P.n_q;
    let h_q = bhq % P.n_q;
    let h_kv = h_q / P.groups;

    var acc: f32 = 0.0;
    for (var d: u32 = 0u; d < P.d_head; d = d + 1u) {
        acc = acc + Q[q_idx(b, h_q, i, d)] * K[kv_idx(b, h_kv, j, d)];
    }
    Scores[s_idx(b, h_q, i, j)] = acc * P.scale;
}

// =============================================================
// Pass 2: row-wise numerically-stable softmax (in place on Scores).
//   One workgroup per row (b, h_q, i); WG_SIZE threads cooperate over j.
// =============================================================
const WG_SOFTMAX : u32 = 256u;

var<workgroup> wg_scratch : array<f32, WG_SOFTMAX>;

@compute @workgroup_size(256, 1, 1)
fn softmax(
    @builtin(workgroup_id)       wg_id : vec3<u32>,
    @builtin(local_invocation_id) lid  : vec3<u32>,
) {
    let bhqi = wg_id.x;
    let total = P.batch * P.n_q * P.seq;
    if (bhqi >= total) { return; }

    let seq = P.seq;
    let i_in_h   = bhqi % seq;
    let bhq      = bhqi / seq;
    let h_q      = bhq  % P.n_q;
    let b        = bhq  / P.n_q;

    let row_base = s_idx(b, h_q, i_in_h, 0u);

    // --- 1. rowmax (per-thread partial, then WG reduce) ---
    let tid = lid.x;
    var m : f32 = -3.4028235e38; // -FLT_MAX
    var j : u32 = tid;
    loop {
        if (j >= seq) { break; }
        let v = Scores[row_base + j];
        if (v > m) { m = v; }
        j = j + WG_SOFTMAX;
    }
    wg_scratch[tid] = m;
    workgroupBarrier();
    // Tree reduce. WG_SOFTMAX is power of two.
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

    // --- 2. exp(x - max), accumulate rowsum ---
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

    // --- 3. normalize ---
    let inv : f32 = 1.0 / row_sum;
    j = tid;
    loop {
        if (j >= seq) { break; }
        Scores[row_base + j] = Scores[row_base + j] * inv;
        j = j + WG_SOFTMAX;
    }
}

// =============================================================
// Pass 3: Probs @ V with GQA broadcasting on V.
//   gid.x = d   (head dim,      0..d_head)
//   gid.y = i   (query pos,     0..seq)
//   gid.z = b*n_q + h_q
// =============================================================
@compute @workgroup_size(16, 16, 1)
fn gqa_pv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d   = gid.x;
    let i   = gid.y;
    let bhq = gid.z;
    if (i >= P.seq || d >= P.d_head) { return; }
    let total_bhq = P.batch * P.n_q;
    if (bhq >= total_bhq) { return; }

    let b   = bhq / P.n_q;
    let h_q = bhq % P.n_q;
    let h_kv = h_q / P.groups;

    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < P.seq; j = j + 1u) {
        acc = acc + Scores[s_idx(b, h_q, i, j)] * V[kv_idx(b, h_kv, j, d)];
    }
    Out[q_idx(b, h_q, i, d)] = acc;
}
