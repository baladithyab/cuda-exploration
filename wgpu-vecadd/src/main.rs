// wgpu vecadd benchmark: c[i] = a[i] + b[i] for f32 vectors, 3-buffer streaming.
// Sweeps N ∈ {1M, 16M, 64M} elements (256M would require >1 GiB per buffer, dropped
// when the adapter cannot grant the storage-buffer limit). Mirrors the cuda-vecadd
// and oxide-vecadd-bench shape: 1 warmup + 10 timed iters per N, GB/s = (3*N*4)/ms,
// correctness probed at idx ∈ {0, n/2, n-1}.
//
// Methodology (Rosetta Stone Wave C1.1 conventions):
//  - GPU-side timing via wgpu::Features::TIMESTAMP_QUERY when available; CPU wall
//    fallback is reported but flagged as inflated by submit/poll overhead.
//  - One pipeline + one bind group per (N) so we don't re-bind across iters.
//  - Naga / WGSL: 256-thread workgroup, 1D dispatch with ceil(N/256) groups.
//  - max_abs_err is computed against host reference for the three probed indices,
//    and a full-buffer readback verifies bit-exactness on the largest run.
//
// Backend note (carried over from wgpu-matmul):
//  On WSL the NVIDIA RTX 5090 is reachable for wgpu only via DX12 + libd3d12, which
//  isn't wired up in this environment. The Vulkan ICD only exposes Mesa llvmpipe
//  (CPU). Numbers will reflect a CPU-side "GPU" run, not the underlying RTX 5090.
//  This is a known limitation of the cross-vendor stack on WSL; the cuda-vecadd
//  and oxide-vecadd cells exercise the real GPU through the CUDA driver path.

use std::time::Instant;
use wgpu::util::DeviceExt;

// 1M, 16M, 64M elems.  256M would need ~1 GiB storage buffers; we keep it for
// "ask if the adapter allows it, otherwise skip" so the bench still runs on the
// llvmpipe default 128 MiB limit.
const SIZES: &[u32] = &[
    1 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
];
const WARMUP: usize = 1;
const ITERS: usize = 10;

fn main() {
    pollster::block_on(run());
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
            "[wgpu] candidate: {} ({:?}, type={:?})",
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
        "[wgpu] using: {} ({:?}, type={:?})",
        info.name, info.backend, info.device_type
    );
    if info.device_type == wgpu::DeviceType::Cpu {
        println!("[wgpu] !! WARNING: only CPU adapter available (WSL Vulkan limitation). Numbers below are CPU-side, not real GPU.");
    }

    let adapter_limits = adapter.limits();
    println!(
        "[wgpu] max_storage_buffer_binding_size = {} MiB",
        adapter_limits.max_storage_buffer_binding_size / (1024 * 1024)
    );
    println!(
        "[wgpu] max_buffer_size               = {} MiB",
        adapter_limits.max_buffer_size / (1024 * 1024)
    );

    // Bytes per buffer for the largest N we want to attempt. Adapters cap
    // request_device() with LimitsExceeded if we ask above what they report
    // (llvmpipe is hard-capped at 128 MiB), so we never request more than
    // the adapter advertises.
    let want_bytes_max = (*SIZES.iter().max().unwrap() as u64) * 4;
    let mut limits = adapter_limits.clone();
    let cap_storage = adapter_limits.max_storage_buffer_binding_size as u64;
    let cap_buffer = adapter_limits.max_buffer_size;
    let req_storage = want_bytes_max.min(cap_storage).min(u32::MAX as u64) as u32;
    let req_buffer = want_bytes_max.min(cap_buffer);
    limits.max_storage_buffer_binding_size = req_storage;
    limits.max_buffer_size = req_buffer;
    println!(
        "[wgpu] requested max_storage_buffer_binding_size = {} MiB (capped by adapter at {} MiB)",
        limits.max_storage_buffer_binding_size / (1024 * 1024),
        cap_storage / (1024 * 1024)
    );

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
    println!("[wgpu] timestamp_query feature: {}", has_ts);

    // Effective device limits (after request_device may clamp).
    let dev_limits = device.limits();
    let max_storage_bytes = dev_limits.max_storage_buffer_binding_size as u64;
    let max_buffer_bytes = dev_limits.max_buffer_size;
    println!(
        "[wgpu] effective max_storage_buffer_binding_size = {} MiB, max_buffer_size = {} MiB",
        max_storage_bytes / (1024 * 1024),
        max_buffer_bytes / (1024 * 1024)
    );

    // Compile shader once.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vecadd"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
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
        entry_point: "vecadd",
        compilation_options: Default::default(),
        cache: None,
    });

    let period = if has_ts {
        queue.get_timestamp_period() as f64
    } else {
        0.0
    };

    let mut summary: Vec<(u32, f64, f64, f64, f64, f32)> = Vec::new();

    for &n_u32 in SIZES {
        let n = n_u32 as usize;
        let nbytes = (n as u64) * 4;
        let traffic_bytes = 3.0 * (n as f64) * 4.0;

        if nbytes > max_storage_bytes || nbytes > max_buffer_bytes {
            println!(
                "[wgpu] N={} ({} MiB/buffer) skipped: exceeds adapter storage-buffer limit ({} MiB)",
                n,
                nbytes / (1024 * 1024),
                max_storage_bytes / (1024 * 1024)
            );
            continue;
        }

        println!(
            "\n[wgpu] N={n} vecadd f32, traffic={:.3} GB/iter, per-buf={} MiB",
            traffic_bytes / 1e9,
            nbytes / (1024 * 1024)
        );

        // Host data, deterministic + matches cuda/oxide vecadd seeds.
        let a: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i % 11) as f32) * 0.01).collect();

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
            size: nbytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_n = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("N"),
            contents: bytemuck::cast_slice(&[n_u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_n.as_entire_binding(),
                },
            ],
        });

        // Timestamp resources — fresh per-N so we don't accidentally hold old data.
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

        let total_wg = (n_u32 + 255) / 256;
        // 2D dispatch to dodge max_compute_workgroups_per_dimension=65535.
        // grid_x * grid_y >= total_wg; pad upward, the in-shader bounds check
        // discards extra threads.
        let max_dim: u32 = 65535;
        let (wg_x, wg_y) = if total_wg <= max_dim {
            (total_wg, 1u32)
        } else {
            let wgx = max_dim;
            let wgy = (total_wg + wgx - 1) / wgx;
            (wgx, wgy)
        };
        println!(
            "[wgpu] N={n} dispatch grid = ({wg_x}, {wg_y}, 1) total_wg={total_wg}"
        );

        // Helper closure: dispatch one kernel + (optional) timestamp pair, return
        // (gpu_ms, cpu_wall_ms).
        let dispatch_once = || {
            let mut enc = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
                pass.dispatch_workgroups(wg_x, wg_y, 1);
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
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    tx.send(r).unwrap();
                });
                device.poll(wgpu::Maintain::Wait);
                rx.recv().unwrap().unwrap();
                let data = slice.get_mapped_range();
                let ts: &[u64] = bytemuck::cast_slice(&data);
                let gpu_ns = (ts[1].wrapping_sub(ts[0])) as f64 * period;
                drop(data);
                ts_read.unmap();
                gpu_ns / 1e6
            } else {
                cpu_ms
            };
            (gpu_ms, cpu_ms)
        };

        for w in 0..WARMUP {
            let (gpu_ms, cpu_ms) = dispatch_once();
            println!("[wgpu] N={n} warmup {w}: gpu={gpu_ms:.3} ms cpu_wall={cpu_ms:.3} ms");
        }

        let mut times_ms: Vec<f64> = Vec::with_capacity(ITERS);
        for i in 0..ITERS {
            let (gpu_ms, cpu_ms) = dispatch_once();
            let gbps = (traffic_bytes / 1e9) / (gpu_ms / 1000.0);
            let ts_label = if has_ts { "gpu_ts" } else { "cpu_wall" };
            println!(
                "[wgpu] N={n} iter={i}: {ts_label}={gpu_ms:.3} ms ({gbps:.2} GB/s)  cpu_wall={cpu_ms:.3} ms"
            );
            times_ms.push(gpu_ms);
        }

        // Correctness: read C back, diff against host reference for full vector.
        // For the largest N this is several hundred MiB but llvmpipe is the slow
        // path anyway; correctness > startup cost.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: nbytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&buf_c, 0, &staging, 0, nbytes);
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let c_host: &[f32] = bytemuck::cast_slice(&data);

        let mut max_abs_err: f32 = 0.0;
        // Sample-based check at the 3 indices the cuda/oxide cells use.
        let probes = [0usize, n / 2, n - 1];
        let mut ok = 0usize;
        for &idx in &probes {
            let got = c_host[idx];
            let want = a[idx] + b[idx];
            let err = (got - want).abs();
            if err == 0.0 {
                ok += 1;
            }
            if err > max_abs_err {
                max_abs_err = err;
            }
            println!(
                "[wgpu] N={n} check idx={idx} got={got} want={want} abs_err={err}"
            );
        }
        // Full-vector max-abs-err pass — vecadd is bit-exact in IEEE-754 single
        // precision, so this should be 0.
        for i in 0..n {
            let want = a[i] + b[i];
            let err = (c_host[i] - want).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
        }
        println!(
            "[wgpu] N={n} correctness: probes {ok}/3 exact, full-vector max_abs_err = {max_abs_err}"
        );
        drop(data);
        staging.unmap();

        // Stats: best + median.
        let mut sorted = times_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let best = sorted[0];
        let median = 0.5 * (sorted[ITERS / 2 - 1] + sorted[ITERS / 2]);
        let best_gb = (traffic_bytes / 1e9) / (best / 1000.0);
        let med_gb = (traffic_bytes / 1e9) / (median / 1000.0);
        println!(
            "[wgpu] N={n} BEST={best:.3} ms ({best_gb:.2} GB/s)  MEDIAN={median:.3} ms ({med_gb:.2} GB/s)"
        );
        summary.push((n_u32, best, median, best_gb, med_gb, max_abs_err));
    }

    println!("\n================== SUMMARY ==================");
    println!(
        "{:<12} {:>12} {:>12} {:>10} {:>10} {:>14}",
        "N", "best_ms", "med_ms", "best_GBs", "med_GBs", "max_abs_err"
    );
    println!("{}", "-".repeat(76));
    for (n, b, m, bg, mg, e) in &summary {
        println!(
            "{:<12} {:>12.4} {:>12.4} {:>10.2} {:>10.2} {:>14}",
            n, b, m, bg, mg, e
        );
    }
    println!("=============================================");
}
