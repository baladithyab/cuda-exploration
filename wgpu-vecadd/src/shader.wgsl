// Naive vecadd: c[i] = a[i] + b[i] for i in 0..n
// 3-buffer streaming kernel: read A, read B, write C.
//
// Workgroup is 256 threads in 1D. We use a 2D dispatch (grid_x, grid_y, 1)
// because wgpu's default max_compute_workgroups_per_dimension is 65535, which
// caps a pure-1D dispatch at ~16.7M threads — not enough for N >= 16M with
// 256-thread workgroups. The flat thread index is reconstructed from the 2D
// grid using num_workgroups.x as the row stride.
@group(0) @binding(0) var<storage, read>       a : array<f32>;
@group(0) @binding(1) var<storage, read>       b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;
@group(0) @binding(3) var<uniform>             n : u32;

@compute @workgroup_size(256, 1, 1)
fn vecadd(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups)        ng:  vec3<u32>,
) {
    // Threads-per-row in the 2D grid = workgroups along x * 256 threads.
    let row_stride: u32 = ng.x * 256u;
    let idx: u32 = gid.y * row_stride + gid.x;
    if (idx >= n) { return; }
    c[idx] = a[idx] + b[idx];
}
