// Two-pass parallel sum reduction in WGSL.
//
// Why two-pass and not atomic-add into one f32?
//   WGSL atomics are restricted to atomic<i32> / atomic<u32>; there is no
//   atomic<f32>. CAS-bit-cast workarounds exist but are slow and fragile.
//   The clean, portable answer: emit one partial per workgroup, then sum
//   the partials with a second dispatch.
//
// Layout:
//   Pass-1 (`reduce_pass1`):
//     workgroup_size = 256, each thread reads ITEMS_PER_THREAD = 4 elements
//     with workgroup-strided access (coalesced inside the WG), accumulates
//     locally, then a classical shared-memory tree reduce ("sequential
//     addressing", no bank conflicts) collapses 256 lanes into one f32.
//     Lane 0 writes partials[wg_id]. Pass-1 grid size = ceil(N / 1024).
//
//   Pass-2 (`reduce_pass2`):
//     1 workgroup of 256 threads. Each thread accumulates a strided slice
//     of the partials buffer (so we can handle any count of partials in
//     one WG), then the same tree reduce produces the final scalar into
//     out[0].
//
// Numerics: f32 throughout, naive +=. CPU reference uses Kahan in f64.
// The 1e-3 abs-err gate at N=16M comfortably exceeds the ~sqrt(N)*eps drift
// of a tree reduce on the (i%7)*0.01 pattern.
//
// Bindings are declared once at module scope; each pipeline only references
// the bindings it needs. `partials` is declared `read_write` so pass-1 can
// store into it; pass-2 only reads it.

struct Params {
    n          : u32,
    n_partials : u32,
};

@group(0) @binding(0) var<storage, read>       data     : array<f32>;
@group(0) @binding(1) var<storage, read_write> partials : array<f32>;
@group(0) @binding(2) var<storage, read_write> out      : array<f32>;
@group(0) @binding(3) var<uniform>              params  : Params;

const BLOCK: u32 = 256u;
const ITEMS_PER_THREAD: u32 = 4u;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn reduce_pass1(
    @builtin(local_invocation_id) lid : vec3<u32>,
    @builtin(workgroup_id)        wid : vec3<u32>,
) {
    let tid = lid.x;
    let n = params.n;
    let block_start = wid.x * (BLOCK * ITEMS_PER_THREAD);

    // Per-thread local accumulate over ITEMS_PER_THREAD elements with
    // workgroup-strided indexing so adjacent threads touch adjacent f32s
    // each iteration -> coalesced loads on real GPUs.
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < ITEMS_PER_THREAD; k = k + 1u) {
        let idx = block_start + k * BLOCK + tid;
        if (idx < n) {
            acc = acc + data[idx];
        }
    }
    sdata[tid] = acc;
    workgroupBarrier();

    // Sequential-addressing tree reduce. At each step the lower half of
    // active lanes adds in the upper half. No shared-mem bank conflicts
    // (vs the older interleaved variant whose stride doubles).
    var s: u32 = BLOCK / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        partials[wid.x] = sdata[0];
    }
}

@compute @workgroup_size(256, 1, 1)
fn reduce_pass2(
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let tid = lid.x;
    let np = params.n_partials;

    var acc: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= np) { break; }
        acc = acc + partials[i];
        i = i + BLOCK;
    }
    sdata[tid] = acc;
    workgroupBarrier();

    var s: u32 = BLOCK / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        out[0] = sdata[0];
    }
}
