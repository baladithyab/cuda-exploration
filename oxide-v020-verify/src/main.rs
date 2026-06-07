/*
 * oxide-v020-verify — Wave V020: cuda-oxide v0.2.0 verification
 *
 * Q1: Does the v0.2.0 typed launch surface (#[cuda_module] + module.<kernel>(...))
 *     obsolete the v0.1.0 gotchas:
 *       #1 artifact-name underscore-vs-hyphen NoArtifact
 *       #3 cuda_launch! stream/module Arc-by-value
 *       #4 &mut DeviceBuffer inner-rebind in helper fns
 * Q2: (separate matmul cell) LDG.E.CONSTANT gap
 * Q3: #[constant] address space — expect PTX .const (AS4), NOT read-only data cache.
 *
 * This cell exercises Q1 (new launch API) and Q3 (#[constant] PTX address space).
 */
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{ConstantMemory, DisjointSlice, constant, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    // Q3: a #[constant] static. Inspect PTX for `.const` / address-space-4.
    #[constant]
    static COEFFS: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;

    // Q3 kernel: reads constant memory.
    #[kernel]
    pub fn poly_const(mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(e) = output.get_mut(idx) {
            let [c0, c1, c2, c3] = COEFFS.get();
            let x = i as f32;
            *e = c0 + c1 * x + c2 * x * x + c3 * x * x * x;
        }
    }

    // Q1 kernel: plain immutable &[f32] input + DisjointSlice output, launched
    // via the typed module surface (no cuda_launch! macro at all).
    #[kernel]
    pub fn saxpy(a: f32, x: &[f32], y: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(e) = out.get_mut(idx) {
            *e = a * x[i] + y[i];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== cuda-oxide v0.2.0 verification (Q1 + Q3) ===\n");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    // ---- Q3: constant memory round-trip ----
    const N: usize = 256;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let coeffs = [1.0f32, 2.0, 0.5, -0.25];
    module.set_coeffs(&stream, &coeffs)?;
    module.poly_const(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev)?;
    let got = out_dev.to_host_vec(&stream)?;
    let want0 = coeffs[0] + coeffs[1] * 3.0 + coeffs[2] * 9.0 + coeffs[3] * 27.0;
    let ok3 = (got[3] - want0).abs() < 1e-2;
    println!("[Q3] #[constant] poly: out[3]={} want={} -> {}", got[3], want0, if ok3 {"OK"} else {"FAIL"});

    // ---- Q1: typed launch with immutable &[f32] args, NO cuda_launch! macro ----
    let n = 1024usize;
    let xh: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let yh: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();
    let x_dev = DeviceBuffer::from_host(&stream, &xh)?;
    let y_dev = DeviceBuffer::from_host(&stream, &yh)?;
    let mut z_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let a = 3.0f32;
    module.saxpy(&stream, LaunchConfig::for_num_elems(n as u32), a, &x_dev, &y_dev, &mut z_dev)?;
    let z = z_dev.to_host_vec(&stream)?;
    let want = a * xh[100] + yh[100];
    let ok1 = (z[100] - want).abs() < 1e-3;
    println!("[Q1] typed saxpy launch: z[100]={} want={} -> {}", z[100], want, if ok1 {"OK"} else {"FAIL"});

    if ok1 && ok3 {
        println!("\nSUCCESS: v0.2.0 typed launch + #[constant] work.");
        Ok(())
    } else {
        Err("verification failed".into())
    }
}
