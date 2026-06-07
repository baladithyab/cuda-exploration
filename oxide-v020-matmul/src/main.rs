/*
 * oxide-v020-matmul — Wave V020 Q2: does v0.2.0 emit LDG.E.CONSTANT for
 * immutable &[f32] matmul operands? (the v0.1.0 finding-b gap)
 *
 * Naive NxN f32 matmul. A,B passed as immutable &[f32]; C as DisjointSlice.
 * After build: disassemble cubin and grep LDG.E.CONSTANT vs LDG.E.
 */
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn matmul(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, thread::Runtime2DIndex>, n: u32) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n as usize) } {
            if row < n as usize {
                let n_size = n as usize;
                let mut acc = 0.0f32;
                let mut i = 0usize;
                while i < n_size {
                    acc += a[row * n_size + i] * b[i * n_size + col];
                    i += 1;
                }
                if let Some(e) = c.get_mut(c_idx) {
                    *e = acc;
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: u32 = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    println!("=== v0.2.0 naive matmul N={} ===", n);
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let sz = (n * n) as usize;
    let ah: Vec<f32> = (0..sz).map(|i| (i % 7) as f32 * 0.1).collect();
    let bh: Vec<f32> = (0..sz).map(|i| (i % 5) as f32 * 0.2).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &ah)?;
    let b_dev = DeviceBuffer::from_host(&stream, &bh)?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, sz)?;

    let cfg = LaunchConfig {
        grid_dim: ((n).div_ceil(16), (n).div_ceil(16), 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    module.matmul(&stream, cfg, &a_dev, &b_dev, &mut c_dev, n)?;
    let c = c_dev.to_host_vec(&stream)?;

    // verify c[0,0] = sum_k a[0,k]*b[k,0]
    let mut want = 0.0f32;
    for k in 0..n as usize { want += ah[k] * bh[k * n as usize]; }
    let ok = (c[0] - want).abs() < 1e-1;
    println!("c[0]={} want={} -> {}", c[0], want, if ok {"OK"} else {"FAIL"});
    if ok { Ok(()) } else { Err("matmul verify failed".into()) }
}
