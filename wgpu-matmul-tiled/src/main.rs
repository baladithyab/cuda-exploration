// wgpu tiled matmul benchmark: C = A * B for f32 NxN matrices.
// Classical shared-memory tiling: 16x16 output tile, BK=16 K-block, one
// thread per output cell.
//
// Methodology mirrors wgpu-matmul/ (the naive baseline) for apples-to-apples:
//  - 1 warmup, 50 timed iterations (Wave C1.3 spec)
//  - Timestamp queries when available (excludes CPU dispatch overhead)
//  - Same input pattern (a[i] = (i%7)*0.01, b[i] = (i%11)*0.01)
//  - 3-cell correctness check vs CPU reference (sampled, not exhaustive,
//    since 16M cells of CPU ref would dominate wall time).
//
// On WSL the only working backend is Vulkan -> llvmpipe (CPU). Real-GPU
// numbers from this folder require running on a non-WSL host.

use std::time::Instant;
use wgpu::util::DeviceExt;

const N: u32 = 4096;
const DEFAULT_ITERS: usize = 50;
const WARMUPS: usize = 1;

fn iters_from_env() -> usize {
    std::env::var("WGPU_TILED_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ITERS)
}

fn main() {
    pollster::block_on(run());
}

fn cpu_ref(a: &[f32], b: &[f32], n: usize, row: usize, col: usize) -> f32 {
    let mut acc: f64 = 0.0;
    for k in 0..n {
        acc += (a[row * n + k] as f64) * (b[k * n + col] as f64);
    }
    acc as f32
}

async fn run() {
    let backends = wgpu::Backends::all();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });

    let mut chosen: Option<wgpu::Adapter> = None;
    let mut fallback: Option<wgpu::Adapter> = None;
    for ad in instance.enumerate_adapters(backends) {
        let info = ad.get_info();
        println!(
            "[wgpu-tiled] candidate: {} ({:?}, type={:?})",
            info.name, info.backend, info.device_type
        );
        if info.device_type != wgpu::DeviceType::Cpu && chosen.is_none() {
            chosen = Some(ad);
        } else if fallback.is_none() {
            fallback = Some(ad);
        }
    }
    let adapter = chosen.or(fallback).expect("no adapter at all");
    let info = adapter.get_info();
    println!(
        "[wgpu-tiled] using: {} ({:?}, type={:?})",
        info.name, info.backend, info.device_type
    );
    if info.device_type == wgpu::DeviceType::Cpu {
        println!("[wgpu-tiled] !! WARNING: only CPU adapter available (WSL Vulkan limitation). Numbers below are CPU-side, not real GPU.");
    }

    let limits = adapter.limits();
    println!(
        "[wgpu-tiled] max_storage_buffer_binding_size = {} MiB",
        limits.max_storage_buffer_binding_size / (1024 * 1024)
    );
    let need = (4096u64 * 4096 * 4) as u32;
    let mut limits = limits;
    if limits.max_storage_buffer_binding_size < need {
        limits.max_storage_buffer_binding_size = need;
        limits.max_buffer_size = limits.max_buffer_size.max(need as u64);
        println!(
            "[wgpu-tiled] requesting max_storage_buffer_binding_size={} MiB",
            need / (1024 * 1024)
        );
    }
    let features_avail = adapter.features();
    let want = wgpu::Features::TIMESTAMP_QUERY;
    let req_features = if features_avail.contains(want) {
        want
    } else {
        wgpu::Features::empty()
    };
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: req_features,
                required_limits: limits,
                ..Default::default()
            },
            None,
        )
        .await
        .expect("device");
    let has_ts = req_features.contains(wgpu::Features::TIMESTAMP_QUERY);
    println!("[wgpu-tiled] timestamp_query feature: {}", has_ts);

    let n = N as usize;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.01).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.01).collect();

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("A"),
        contents: bytemuck::cast_slice(&a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("B"),
        contents: bytemuck::cast_slice(&b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C"),
        size: (n * n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_n = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dim"),
        contents: bytemuck::cast_slice(&[N]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul-tiled"),
        source: wgpu::ShaderSource::Wgsl(include_str!("matmul.wgsl").into()),
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "matmul",
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_n.as_entire_binding() },
        ],
    });

    // Timestamp query
    let ts_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("ts"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let ts_resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_resolve"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let ts_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_read"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Readback buffer for correctness check
    let buf_c_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C_read"),
        size: (n * n * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let wg = (N + 15) / 16;
    let total_flops = 2.0_f64 * (n as f64).powi(3);
    println!(
        "[wgpu-tiled] matmul {}x{} f32 (BM=BN=BK=16), {:.2} GFLOP/iter",
        N, N,
        total_flops / 1e9
    );

    let dispatch_once = |label: &str, iter: usize, has_ts: bool| -> (f64, f64) {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let timestamp_writes = if has_ts {
                Some(wgpu::ComputePassTimestampWrites {
                    query_set: &ts_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                })
            } else {
                None
            };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg, wg, 1);
        }
        if has_ts {
            enc.resolve_query_set(&ts_set, 0..2, &ts_resolve, 0);
            enc.copy_buffer_to_buffer(&ts_resolve, 0, &ts_read, 0, 16);
        }
        let cpu_start = Instant::now();
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        let gpu_ms = if has_ts {
            let slice = ts_read.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let ts: &[u64] = bytemuck::cast_slice(&data);
            let period = queue.get_timestamp_period() as f64;
            let gpu_ns = (ts[1].wrapping_sub(ts[0])) as f64 * period;
            drop(data);
            ts_read.unmap();
            gpu_ns / 1e6
        } else {
            cpu_ms
        };

        let tflops = (total_flops / 1e12) / (gpu_ms / 1000.0);
        let ts_label = if has_ts { "gpu_ts" } else { "cpu_wall" };
        println!(
            "[wgpu-tiled] {label} {iter}: {ts_label}={gpu_ms:.2} ms ({tflops:.3} TFLOPS)  cpu_wall={cpu_ms:.2} ms"
        );
        (gpu_ms, cpu_ms)
    };

    let iters = iters_from_env();
    println!(
        "[wgpu-tiled] timed iters: {iters} (override via WGPU_TILED_ITERS, default {DEFAULT_ITERS})"
    );

    // Warmup
    for w in 0..WARMUPS {
        let _ = dispatch_once("warmup", w, has_ts);
    }

    // Timed iterations
    let mut times_ms: Vec<f64> = Vec::with_capacity(iters);
    for it in 0..iters {
        let (gpu_ms, _) = dispatch_once("iter", it, has_ts);
        times_ms.push(gpu_ms);
    }

    // Correctness check: sample 5 cells, compare against CPU ref.
    // Copy C back.
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&buf_c, 0, &buf_c_read, 0, (n * n * 4) as u64);
    queue.submit(Some(enc.finish()));
    device.poll(wgpu::Maintain::Wait);
    let slice = buf_c_read.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let c: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buf_c_read.unmap();

    let pts: [(usize, usize); 5] = [
        (0, 0),
        (1, 1),
        (n / 2, n / 2),
        (n - 1, 0),
        (n - 1, n - 1),
    ];
    let mut max_abs_err: f32 = 0.0;
    let mut max_rel_err: f32 = 0.0;
    let mut ok = 0;
    for &(r, col) in &pts {
        let got = c[r * n + col];
        let want = cpu_ref(&a, &b, n, r, col);
        let abs_err = (got - want).abs();
        let rel_err = abs_err / want.abs().max(1e-6);
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);
        let tag = if rel_err < 1e-3 { "OK" } else { "FAIL" };
        if rel_err < 1e-3 {
            ok += 1;
        }
        println!(
            "[wgpu-tiled] check ({r},{col}): got={got:.4} want={want:.4} abs={abs_err:.3e} rel={rel_err:.3e} {tag}"
        );
    }
    println!(
        "[wgpu-tiled] correctness: {ok}/5 OK  max_abs_err={max_abs_err:.3e}  max_rel_err={max_rel_err:.3e}"
    );

    times_ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let best = times_ms[0];
    let median = times_ms[times_ms.len() / 2];
    let best_tf = (total_flops / 1e12) / (best / 1000.0);
    let med_tf = (total_flops / 1e12) / (median / 1000.0);
    println!("\n[wgpu-tiled] BEST   {best:.2} ms  {best_tf:.3} TFLOPS");
    println!("[wgpu-tiled] MEDIAN {median:.2} ms  {med_tf:.3} TFLOPS");
}
