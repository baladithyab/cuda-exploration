// Classical shared-memory tiled matmul: C = A * B for f32 NxN matrices.
// One thread per output cell. Workgroup is 16x16 = 256 threads.
// Per workgroup: walk K in tiles of BK=16, each tile cooperatively loaded
// into workgroup-shared memory, then accumulated in a per-thread register.
//
// Layout: BM = BN = BK = 16. Each WG produces a 16x16 output tile. Each thread
// handles exactly one A tile load and one B tile load per K-tile (256 threads,
// 16x16 = 256 elements per tile -> perfect 1:1 mapping).
//
// Pitfall: WGSL var<workgroup> arrays must be sized with a literal expression
// that the validator can fold to a constant. We use `array<f32, 256>` directly
// (matching BM*BK = BK*BN = 256). Trying to size from a `const TILE: u32 = ...`
// works in wgpu 22 but the safer cross-validator path is the literal.

@group(0) @binding(0) var<storage, read>       a   : array<f32>;
@group(0) @binding(1) var<storage, read>       b   : array<f32>;
@group(0) @binding(2) var<storage, read_write> c   : array<f32>;
@group(0) @binding(3) var<uniform>             dim : u32;

var<workgroup> tile_a : array<f32, 256>;
var<workgroup> tile_b : array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn matmul(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id)  lid : vec3<u32>,
    @builtin(workgroup_id)         wid : vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let tx  = lid.x;        // 0..15  (column within tile)
    let ty  = lid.y;        // 0..15  (row within tile)
    let local = ty * 16u + tx;

    let n = dim;
    // dim is a multiple of 16 in our sweep (4096), so guard here is structural.
    let in_bounds = (row < n) & (col < n);

    var acc : f32 = 0.0;
    let num_tiles = n / 16u;

    for (var t : u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load A[row, t*16 + tx] into tile_a[ty, tx]
        let a_col = t * 16u + tx;
        let b_row = t * 16u + ty;

        // Each thread loads exactly one element of each tile.
        // For our N=4096 sweep all indices are in-bounds.
        if (row < n && a_col < n) {
            tile_a[local] = a[row * n + a_col];
        } else {
            tile_a[local] = 0.0;
        }
        if (b_row < n && col < n) {
            tile_b[local] = b[b_row * n + col];
        } else {
            tile_b[local] = 0.0;
        }

        workgroupBarrier();

        // Inner K loop: each thread reads tile_a[ty, k] and tile_b[k, tx]
        // and accumulates 16 products. Naga should unroll this; manual unroll
        // is left for tuning.
        for (var k : u32 = 0u; k < 16u; k = k + 1u) {
            acc = acc + tile_a[ty * 16u + k] * tile_b[k * 16u + tx];
        }

        workgroupBarrier();
    }

    if (in_bounds) {
        c[row * n + col] = acc;
    }
}
