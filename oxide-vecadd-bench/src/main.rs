// Wave 4 W4B: memory-bandwidth bench in cuda-oxide.
// Two kernels: `vecadd_safe` (slice indexing, bounds-checked) and
// `vecadd_unchecked` (raw pointer, no bounds check). Block 256 (via
// LaunchConfig::for_num_elems). Sweep N ∈ {1M, 16M, 64M, 256M} f32.
// cuEvent timing per ADR-0001 (same shape as oxide-matmul::run_kernel_sweep).

use cuda_core::{CudaContext, CudaEvent, CudaModule, CudaStream, DeviceBuffer, LaunchConfig, sys};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

const N_MAX: usize = 256 * 1024 * 1024;
const SIZES: [usize; 4] = [
    1 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
];
const WARMUP: usize = 1;
const ITERS: usize = 10;

#[kernel]
pub fn vecadd_safe(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}

#[kernel]
pub fn vecadd_unchecked(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, n: u32) {
    let idx = thread::index_1d().get();
    if idx >= n as usize {
        return;
    }
    // SAFETY: idx < n ≤ buffer capacity for all three buffers.
    unsafe {
        let av = *a.as_ptr().add(idx);
        let bv = *b.as_ptr().add(idx);
        *c.as_mut_ptr().add(idx) = av + bv;
    }
}

#[allow(clippy::too_many_arguments)]
fn run_kernel_sweep(
    kernel_name: &str,
    n: usize,
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    a_dev: &DeviceBuffer<f32>,
    b_dev: &DeviceBuffer<f32>,
    mut c_dev: &mut DeviceBuffer<f32>,
    use_unchecked: bool,
    csv: &mut BufWriter<File>,
) -> Vec<(f64, f64)> {
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let n_arg = n as u32;
    let traffic_bytes: f64 = 3.0 * (n as f64) * 4.0;

    for _ in 0..WARMUP {
        let s = stream.clone();
        let m = module.clone();
        if use_unchecked {
            cuda_launch! {
                kernel: vecadd_unchecked, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), n_arg]
            }
            .unwrap();
        } else {
            cuda_launch! {
                kernel: vecadd_safe, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
            }
            .unwrap();
        }
        stream.synchronize().unwrap();
    }

    let mut results: Vec<(f64, f64)> = Vec::with_capacity(ITERS);
    for i in 0..ITERS {
        let start: CudaEvent = ctx
            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .expect("new_event start");
        let stop: CudaEvent = ctx
            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .expect("new_event stop");

        let s = stream.clone();
        let m = module.clone();

        let t0 = Instant::now();
        start.record(stream).expect("record start");
        if use_unchecked {
            cuda_launch! {
                kernel: vecadd_unchecked, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), n_arg]
            }
            .unwrap();
        } else {
            cuda_launch! {
                kernel: vecadd_safe, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
            }
            .unwrap();
        }
        stop.record(stream).expect("record stop");
        stream.synchronize().unwrap();
        let _cpu_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let gpu_ms = start.elapsed_ms(&stop).expect("elapsed_ms") as f64;
        let gbps = (traffic_bytes / 1e9) / (gpu_ms / 1000.0);

        println!("[oxide-{kernel_name}] N={n} iter={i} gpu_ms={gpu_ms:.4} GB/s={gbps:.2}");
        writeln!(
            csv,
            "oxide,{kernel_name},{n},{i},{gpu_ms:.6},{gbps:.6}"
        )
        .expect("csv write");
        results.push((gpu_ms, gbps));
    }
    results
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_vecadd_bench").expect("load");

    let csv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/results.csv");
    let csv_file = File::create(csv_path).expect("create results.csv");
    let mut csv = BufWriter::new(csv_file);
    writeln!(csv, "impl,kernel,N_elems,iter,gpu_ms,GB_per_s").unwrap();

    println!(
        "[oxide] vecadd 3-buffer streaming sweep {:?}, {} warmup + {} iters",
        SIZES, WARMUP, ITERS
    );

    // Allocate buffers ONCE for N_MAX. Populate with max-sized host data; kernel
    // only touches the first n elements per iter.
    let a_host_max: Vec<f32> = (0..N_MAX).map(|i| ((i % 7) as f32) * 0.01).collect();
    let b_host_max: Vec<f32> = (0..N_MAX).map(|i| ((i % 11) as f32) * 0.01).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a_host_max).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host_max).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N_MAX).unwrap();

    let mut summary: Vec<(String, usize, f64, f64, f64, f64)> = Vec::new();

    for &n in &SIZES {
        println!("---- N = {n} ----");
        for &(kname, use_unchk) in &[("safe", false), ("unchecked", true)] {
            let iters = run_kernel_sweep(
                kname,
                n,
                &ctx,
                &stream,
                &module,
                &a_dev,
                &b_dev,
                &mut c_dev,
                use_unchk,
                &mut csv,
            );
            let mut gpu_times: Vec<f64> = iters.iter().map(|(g, _)| *g).collect();
            gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let best = gpu_times[0];
            let med = 0.5 * (gpu_times[ITERS / 2 - 1] + gpu_times[ITERS / 2]);
            let traffic: f64 = 3.0 * (n as f64) * 4.0;
            let best_gb = (traffic / 1e9) / (best / 1000.0);
            let med_gb = (traffic / 1e9) / (med / 1000.0);
            summary.push((kname.to_string(), n, best, med, best_gb, med_gb));

            // Correctness at 0, n/2, n-1. Download just the three values.
            let c_host_all = c_dev.to_host_vec(&stream).unwrap();
            let checks = [0usize, n / 2, n - 1];
            let mut ok = 0;
            for &idx in &checks {
                let got = c_host_all[idx];
                let want = a_host_max[idx] + b_host_max[idx];
                let rel = (got - want).abs() / want.abs().max(1e-6);
                if rel < 1e-3 {
                    ok += 1;
                }
            }
            println!(
                "[oxide-{kname}] N={n} correctness {ok}/3  best={best:.4}ms med={med:.4}ms  ({med_gb:.2} GB/s median)"
            );
        }
        println!();
    }

    csv.flush().unwrap();

    println!("================== SUMMARY ==================");
    println!(
        "{:<12} {:>10} {:>12} {:>12} {:>10} {:>10}",
        "kernel", "N", "best_gpu_ms", "med_gpu_ms", "best_GBs", "med_GBs"
    );
    println!("{}", "-".repeat(72));
    for (k, n, b, m, bg, mg) in &summary {
        println!(
            "{:<12} {:>10} {:>12.4} {:>12.4} {:>10.2} {:>10.2}",
            k, n, b, m, bg, mg
        );
    }
    println!("=============================================");
}
