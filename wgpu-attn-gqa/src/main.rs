// Wave 16.2 — wgpu/WGSL naive 3-pass Grouped-Query Attention.
//
// **Honest note on WSL2 wgpu → NVIDIA:** per waves 1-11 (see wgpu-matmul/ANALYSIS.md
// and skills/mlops/rust-gpu-compute/SKILL.md), on this WSL2 host the only adapters
// wgpu sees are llvmpipe software rasterisers (Vulkan + GL), and the DX12 backend
// cannot enumerate the GPU without Microsoft's Dozen ICD. So this program is
// *expected* to run on CPU via llvmpipe — that is the headline finding, not a
// performance number. The WGSL kernel is still correct and will run on a real GPU
// on bare-metal Linux with libnvidia-vulkan, on Windows via DX12, or in a browser
// via WebGPU.
//
// If the WSL2 limitation has been lifted since (driver update, dozen install),
// the enumerate-adapters loop will pick the real GPU automatically and the bench
// numbers will be meaningful. The code does not treat CPU adapters as a hard
// error — it runs, warns loudly, and the logged adapter type is the truth.
//
// Two shapes:
//   correctness  (1, 4, 128, 64)  / Nkv=2 — numerical correctness vs PyTorch reference
//   llama3_8b    (1, 32, 2048, 128) / Nkv=8 — perf shape
//
// FLOPs per forward (from flops.py):
//   2 * B * Nq * S * S * D    (QK^T)
//   + ~5 * B * Nq * S * S     (softmax, small)
//   + 2 * B * Nq * S * S * D  (PV)
// Simplified to 4 * B * Nq * S^2 * D for the headline TFLOPS (softmax ignored).

use std::path::PathBuf;
use std::time::Instant;
use wgpu::util::DeviceExt;

// ---------- shape defs (mirrors reference/shapes.py) ----------
#[derive(Debug, Clone, Copy)]
struct Shape {
    name: &'static str,
    batch: u32,
    seq: u32,
    n_q: u32,
    n_kv: u32,
    d_head: u32,
}

impl Shape {
    fn groups(&self) -> u32 {
        self.n_q / self.n_kv
    }
    fn scale(&self) -> f32 {
        1.0_f32 / (self.d_head as f32).sqrt()
    }
    fn q_elems(&self) -> usize {
        (self.batch * self.n_q * self.seq * self.d_head) as usize
    }
    fn kv_elems(&self) -> usize {
        (self.batch * self.n_kv * self.seq * self.d_head) as usize
    }
    fn scores_elems(&self) -> usize {
        (self.batch * self.n_q * self.seq * self.seq) as usize
    }
    fn flops(&self) -> f64 {
        // 2 * B * Nq * S^2 * D (QKt) + 2 * B * Nq * S^2 * D (PV) = 4 * B * Nq * S^2 * D
        let (b, nq, s, d) = (self.batch as f64, self.n_q as f64, self.seq as f64, self.d_head as f64);
        4.0 * b * nq * s * s * d
    }
    /// per-dtype tolerance; matches reference/tolerances.py f32 row
    fn atol(&self) -> f32 { 1e-5 }
    fn rtol(&self) -> f32 { 1e-5 }
}

const CORRECTNESS: Shape = Shape { name: "correctness", batch: 1, seq: 128,  n_q: 4,  n_kv: 2, d_head: 64 };
// `medium_512` is NOT a canonical wave15 shape — no .npy inputs exist for it and its
// output isn't checked. It exists only as a perf-sanity intermediate for the WSL2
// llvmpipe run, since the canonical llama3_8b bench shape needs a 512 MiB scores buffer
// which exceeds llvmpipe's 128 MiB max_storage_buffer_binding_size. We synthesize Q/K/V
// with a deterministic pattern. On a real GPU with >=512 MiB storage bindings, uncomment
// LLAMA3_8B and wire it up the same way.
const MEDIUM_512: Shape = Shape { name: "medium_512",  batch: 1, seq: 512,  n_q: 8,  n_kv: 2, d_head: 64 };
const LLAMA3_8B:   Shape = Shape { name: "llama3_8b",   batch: 1, seq: 2048, n_q: 32, n_kv: 8, d_head: 128 };

// ---------- .npy loader (f32 only, C-contiguous, no fortran-order support) ----------
fn load_npy_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {:?}: {e}", path));
    let reader = npyz::NpyFile::new(&bytes[..]).expect("parse npy header");
    // dtype check
    let header = reader.header().clone();
    let dtype_ok = matches!(
        header.dtype(),
        npyz::DType::Plain(ref s) if {
            let raw = s.to_string();
            raw.contains("f4") || raw.contains("float32")
        }
    );
    if !dtype_ok {
        panic!("{:?} not f32: dtype={:?}", path, header.dtype());
    }
    reader.into_vec::<f32>().expect("read f32")
}

// ---------- uniform buffer layout (matches WGSL Params) ----------
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    batch:  u32,
    n_q:    u32,
    n_kv:   u32,
    seq:    u32,
    d_head: u32,
    groups: u32,
    _pad0:  u32,
    _pad1:  u32,
    scale:  f32,
    _pad2:  f32,
    _pad3:  f32,
    _pad4:  f32,
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    // =====================================================================
    // 1. Adapter enumeration — honest and loud about CPU-only fallback.
    // =====================================================================
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
            "[wgpu] candidate: {:<50} backend={:?}  type={:?}",
            info.name, info.backend, info.device_type
        );
        if info.device_type != wgpu::DeviceType::Cpu && chosen.is_none() {
            chosen = Some(ad);
        } else if fallback.is_none() {
            fallback = Some(ad);
        }
    }
    let adapter = chosen.or(fallback).expect("no adapter found at all");
    let info = adapter.get_info();
    let is_cpu = info.device_type == wgpu::DeviceType::Cpu;
    println!(
        "\n[wgpu] using: {} ({:?}, type={:?})",
        info.name, info.backend, info.device_type
    );
    if is_cpu {
        println!("[wgpu] !! WARNING: only CPU adapters enumerated. This is the documented");
        println!("[wgpu] !! WSL2 dead-end (Wave 1-11). Kernel will run via llvmpipe on CPU.");
        println!("[wgpu] !! Any TFLOPS number below is NOT GPU perf; it is llvmpipe throughput.");
        println!("[wgpu] !! Correctness is still meaningful; timing is not.");
    } else {
        println!("[wgpu] GPU adapter available — numbers below reflect real hardware.");
    }

    // =====================================================================
    // 2. Device + queue. Request raised limits for llama3_8b and timestamps if offered.
    // =====================================================================
    let features_avail = adapter.features();
    let want_ts = wgpu::Features::TIMESTAMP_QUERY;
    let req_features = if features_avail.contains(want_ts) { want_ts } else { wgpu::Features::empty() };

    // llama3_8b scores buffer = 1*32*2048*2048*4 = 512 MiB.  V buffer = 32 MiB. We would like
    // to raise limits, but wgpu rejects requests that exceed the adapter's reported cap.
    // So we clamp to what the adapter offers; if scores > cap, the bench shape will SKIP later.
    let adapter_limits = adapter.limits();
    let mut limits = adapter_limits.clone();
    // Keep defaults from adapter; only raise invocations-per-wg if adapter supports more.
    limits.max_compute_invocations_per_workgroup =
        limits.max_compute_invocations_per_workgroup.max(256);

    println!(
        "[wgpu] adapter caps: max_storage_buffer_binding_size = {} MiB, max_buffer_size = {} MiB",
        adapter_limits.max_storage_buffer_binding_size / (1024 * 1024),
        adapter_limits.max_buffer_size / (1024 * 1024),
    );

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("wgpu-attn-gqa"),
                required_features: req_features,
                required_limits: limits,
                ..Default::default()
            },
            None,
        )
        .await
        .expect("device");
    let has_ts = req_features.contains(wgpu::Features::TIMESTAMP_QUERY);
    println!("[wgpu] timestamp_query feature: {has_ts}\n");

    // =====================================================================
    // 3. Build pipelines once (shared across both shapes).
    // =====================================================================
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("attn.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("attn.wgsl").into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("attn-bgl"),
        entries: &[
            // 0: Params uniform
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 1-3: Q, K, V read-only storage
            bgl_storage_entry(1, true),
            bgl_storage_entry(2, true),
            bgl_storage_entry(3, true),
            // 4: Scores read-write
            bgl_storage_entry(4, false),
            // 5: Out read-write
            bgl_storage_entry(5, false),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("attn-pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let make_pipe = |entry: &'static str| -> wgpu::ComputePipeline {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: entry, // wgpu 22: &'a str, not Option<&str>
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pipe_qkt = make_pipe("gqa_qkt");
    let pipe_sm  = make_pipe("softmax");
    let pipe_pv  = make_pipe("gqa_pv");

    // =====================================================================
    // 4. Run correctness shape first (must pass before we bother with bench).
    // =====================================================================
    let inputs_dir: PathBuf = PathBuf::from(
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs",
    );
    println!("[wgpu] inputs dir = {:?}\n", inputs_dir);

    println!("=== SHAPE: correctness ===");
    let correctness_ok = run_shape(
        &device, &queue, &bgl, &pipe_qkt, &pipe_sm, &pipe_pv,
        &CORRECTNESS, &inputs_dir, /*iters=*/ 2, has_ts,
    ).await;

    if !correctness_ok {
        println!("\n[wgpu] correctness FAILED — skipping bench shape.");
        return;
    }

    println!("\n=== SHAPE: medium_512 (perf-only, synthetic inputs, no correctness check) ===");
    println!("[medium_512] note: canonical bench llama3_8b needs 512 MiB scores buffer but");
    println!("[medium_512]       llvmpipe caps at {} MiB. medium_512 is a scaled-down perf sanity.",
        device.limits().max_storage_buffer_binding_size / (1024*1024));
    let medium_need = (MEDIUM_512.scores_elems() as u64) * 4;
    if medium_need > device.limits().max_storage_buffer_binding_size as u64 {
        println!("[medium_512] SKIP: scores {} MiB > max {} MiB",
            medium_need / (1024*1024),
            device.limits().max_storage_buffer_binding_size / (1024*1024));
    } else {
        let _ = run_shape_synth(
            &device, &queue, &bgl, &pipe_qkt, &pipe_sm, &pipe_pv,
            &MEDIUM_512, /*iters=*/ 3, has_ts,
        ).await;
    }

    println!("\n=== SHAPE: llama3_8b (canonical bench) ===");
    let need = (LLAMA3_8B.scores_elems() as u64) * 4;
    if need > device.limits().max_storage_buffer_binding_size as u64 {
        println!("[llama3_8b] SKIP: scores buffer {} MiB exceeds max_storage_buffer_binding_size {} MiB",
            need / (1024*1024),
            device.limits().max_storage_buffer_binding_size / (1024*1024));
        println!("[llama3_8b] This is a wgpu-portable-limit issue: the default max storage binding");
        println!("[llama3_8b] (128 MiB on llvmpipe, often 2 GiB on NVIDIA Vulkan) bounds seq^2 * n_q.");
        println!("[llama3_8b] To run this shape would require either (a) a real GPU adapter with");
        println!("[llama3_8b] a higher binding cap, or (b) splitting scores into per-head buffers");
        println!("[llama3_8b] (which is how fused/FlashAttention avoids the problem entirely).");
    } else {
        let _ = run_shape(
            &device, &queue, &bgl, &pipe_qkt, &pipe_sm, &pipe_pv,
            &LLAMA3_8B, &inputs_dir, /*iters=*/ 4, has_ts,
        ).await;
    }

    if is_cpu {
        println!("\n[wgpu] Reminder: these times are llvmpipe on CPU. NOT GPU perf.");
    }
}

fn bgl_storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

async fn run_shape(
    device: &wgpu::Device,
    queue:  &wgpu::Queue,
    bgl:    &wgpu::BindGroupLayout,
    pipe_qkt: &wgpu::ComputePipeline,
    pipe_sm:  &wgpu::ComputePipeline,
    pipe_pv:  &wgpu::ComputePipeline,
    shape:  &Shape,
    inputs_dir: &PathBuf,
    iters:  usize,
    has_ts: bool,
) -> bool {
    println!("[{}] B={} Nq={} Nkv={} S={} D={} groups={}",
        shape.name, shape.batch, shape.n_q, shape.n_kv, shape.seq, shape.d_head, shape.groups());

    // ----- load Q,K,V (f32) + expected output -----
    let q = load_npy_f32(&inputs_dir.join(format!("gqa_{}_q_f32.npy", shape.name)));
    let k = load_npy_f32(&inputs_dir.join(format!("gqa_{}_k_f32.npy", shape.name)));
    let v = load_npy_f32(&inputs_dir.join(format!("gqa_{}_v_f32.npy", shape.name)));
    let expected = load_npy_f32(&inputs_dir.join(format!("gqa_{}_expected_f32.npy", shape.name)));
    assert_eq!(q.len(), shape.q_elems());
    assert_eq!(k.len(), shape.kv_elems());
    assert_eq!(v.len(), shape.kv_elems());
    assert_eq!(expected.len(), shape.q_elems());
    println!("[{}] loaded Q ({:.1} MiB)  K+V ({:.1} MiB ea)  expected ({:.1} MiB)",
        shape.name,
        q.len() as f64 * 4.0 / (1024.0 * 1024.0),
        k.len() as f64 * 4.0 / (1024.0 * 1024.0),
        expected.len() as f64 * 4.0 / (1024.0 * 1024.0),
    );

    // ----- params uniform -----
    let params = Params {
        batch:  shape.batch,
        n_q:    shape.n_q,
        n_kv:   shape.n_kv,
        seq:    shape.seq,
        d_head: shape.d_head,
        groups: shape.groups(),
        _pad0: 0, _pad1: 0,
        scale:  shape.scale(),
        _pad2: 0.0, _pad3: 0.0, _pad4: 0.0,
    };

    let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buf_q = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Q"), contents: bytemuck::cast_slice(&q),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_k = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("K"), contents: bytemuck::cast_slice(&k),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("V"), contents: bytemuck::cast_slice(&v),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let scores_bytes = (shape.scores_elems() * 4) as u64;
    let out_bytes    = (shape.q_elems() * 4) as u64;
    let buf_scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scores"),
        size: scores_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buf_scores.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: buf_out.as_entire_binding() },
        ],
    });

    // Timestamp setup — 6 queries: qkt_begin, qkt_end, sm_begin, sm_end, pv_begin, pv_end
    let ts_count: u32 = 6;
    let ts_set = if has_ts {
        Some(device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("ts"),
            ty: wgpu::QueryType::Timestamp,
            count: ts_count,
        }))
    } else { None };
    let ts_resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_resolve"),
        size: (ts_count as u64) * 8,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let ts_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_read"),
        size: (ts_count as u64) * 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let ts_period = if has_ts { queue.get_timestamp_period() as f64 } else { 0.0 };

    // Dispatch geometry
    let wg_qkt = ((shape.seq + 15) / 16, (shape.seq + 15) / 16, shape.batch * shape.n_q);
    let wg_sm  = (shape.batch * shape.n_q * shape.seq, 1, 1);
    let wg_pv  = ((shape.d_head + 15) / 16, (shape.seq + 15) / 16, shape.batch * shape.n_q);

    let total_flops = shape.flops();
    println!("[{}] flops/iter = {:.2} GFLOP   dispatches: qkt=({}x{}x{}) sm=({}x{}x{}) pv=({}x{}x{})",
        shape.name, total_flops / 1e9,
        wg_qkt.0, wg_qkt.1, wg_qkt.2,
        wg_sm.0, wg_sm.1, wg_sm.2,
        wg_pv.0, wg_pv.1, wg_pv.2);

    let mut total_ms_vec: Vec<f64> = Vec::new();
    let mut qkt_ms_vec:   Vec<f64> = Vec::new();
    let mut sm_ms_vec:    Vec<f64> = Vec::new();
    let mut pv_ms_vec:    Vec<f64> = Vec::new();

    for iter in 0..(iters + 1) {
        let is_warmup = iter == 0;
        let label = if is_warmup { "warmup" } else { "iter" };

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // --- QKt pass ---
        {
            let timestamp_writes = ts_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            });
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("qkt"), timestamp_writes });
            p.set_pipeline(pipe_qkt);
            p.set_bind_group(0, &bind_group, &[]);
            p.dispatch_workgroups(wg_qkt.0, wg_qkt.1, wg_qkt.2);
        }
        // --- softmax pass ---
        {
            let timestamp_writes = ts_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(2),
                end_of_pass_write_index: Some(3),
            });
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("softmax"), timestamp_writes });
            p.set_pipeline(pipe_sm);
            p.set_bind_group(0, &bind_group, &[]);
            p.dispatch_workgroups(wg_sm.0, wg_sm.1, wg_sm.2);
        }
        // --- PV pass ---
        {
            let timestamp_writes = ts_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(4),
                end_of_pass_write_index: Some(5),
            });
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pv"), timestamp_writes });
            p.set_pipeline(pipe_pv);
            p.set_bind_group(0, &bind_group, &[]);
            p.dispatch_workgroups(wg_pv.0, wg_pv.1, wg_pv.2);
        }
        if let Some(qs) = ts_set.as_ref() {
            enc.resolve_query_set(qs, 0..ts_count, &ts_resolve, 0);
            enc.copy_buffer_to_buffer(&ts_resolve, 0, &ts_read, 0, (ts_count as u64) * 8);
        }
        // copy Out -> readback only on last iter (cheap), but we do it every iter for simplicity
        if iter == iters {
            enc.copy_buffer_to_buffer(&buf_out, 0, &buf_readback, 0, out_bytes);
        }

        let cpu_start = Instant::now();
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        // Read timestamps
        let (qkt_ms, sm_ms, pv_ms, total_ms) = if has_ts {
            let slice = ts_read.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let ts: &[u64] = bytemuck::cast_slice(&data);
            let ns = |a: u64, b: u64| (b.wrapping_sub(a)) as f64 * ts_period;
            let r = (ns(ts[0], ts[1])/1e6, ns(ts[2], ts[3])/1e6, ns(ts[4], ts[5])/1e6,
                     ns(ts[0], ts[5])/1e6);
            drop(data);
            ts_read.unmap();
            r
        } else {
            // Fall back to cpu-wall as total; per-pass unknown.
            (f64::NAN, f64::NAN, f64::NAN, cpu_ms)
        };

        let tflops = (total_flops / 1e12) / (total_ms / 1000.0);
        println!(
            "[{}] {} {}:  total={:.3} ms  qkt={:.3} sm={:.3} pv={:.3}  ({:.4} TFLOPS)  cpu_wall={:.2} ms",
            shape.name, label, iter, total_ms, qkt_ms, sm_ms, pv_ms, tflops, cpu_ms,
        );
        if !is_warmup {
            total_ms_vec.push(total_ms);
            if !qkt_ms.is_nan() { qkt_ms_vec.push(qkt_ms); sm_ms_vec.push(sm_ms); pv_ms_vec.push(pv_ms); }
        }
    }

    // ----- correctness check (from last iter's readback) -----
    let slice = buf_readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
    drop(data);
    buf_readback.unmap();

    let (max_abs, rel, ok) = check_correctness(&out, &expected, shape.atol(), shape.rtol());

    // ----- summary -----
    total_ms_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best  = total_ms_vec.first().copied().unwrap_or(f64::NAN);
    let med   = total_ms_vec.get(total_ms_vec.len()/2).copied().unwrap_or(f64::NAN);
    let best_tf = (total_flops / 1e12) / (best / 1000.0);
    let med_tf  = (total_flops / 1e12) / (med / 1000.0);
    println!(
        "[{}] BEST   total={:.3} ms  {:.4} TFLOPS",
        shape.name, best, best_tf,
    );
    println!(
        "[{}] MEDIAN total={:.3} ms  {:.4} TFLOPS",
        shape.name, med, med_tf,
    );
    if !qkt_ms_vec.is_empty() {
        qkt_ms_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sm_ms_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pv_ms_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let s = total_ms_vec.iter().sum::<f64>() / total_ms_vec.len() as f64;
        println!("[{}] per-stage median:  qkt={:.3} ms ({:.1}%)  sm={:.3} ms ({:.1}%)  pv={:.3} ms ({:.1}%)",
            shape.name,
            qkt_ms_vec[qkt_ms_vec.len()/2], qkt_ms_vec[qkt_ms_vec.len()/2] / s * 100.0,
            sm_ms_vec[sm_ms_vec.len()/2],   sm_ms_vec[sm_ms_vec.len()/2]   / s * 100.0,
            pv_ms_vec[pv_ms_vec.len()/2],   pv_ms_vec[pv_ms_vec.len()/2]   / s * 100.0,
        );
    }
    println!("[{}] correctness: max_abs_err={:.3e}  rel_err={:.3e}  atol={:.0e}  {}",
        shape.name, max_abs, rel, shape.atol(),
        if ok { "OK" } else { "FAIL" });
    ok
}

fn check_correctness(got: &[f32], exp: &[f32], atol: f32, rtol: f32) -> (f32, f32, bool) {
    assert_eq!(got.len(), exp.len());
    let mut max_abs: f32 = 0.0;
    let mut max_exp: f32 = 0.0;
    let mut ok = true;
    for (a, b) in got.iter().zip(exp.iter()) {
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
        if b.abs() > max_exp { max_exp = b.abs(); }
        let tol = atol + rtol * b.abs();
        if d > tol || !a.is_finite() { ok = false; }
    }
    let rel = max_abs / (max_exp + 1e-30);
    (max_abs, rel, ok)
}

// Same harness as run_shape, but synthesizes Q/K/V deterministically instead of loading
// from .npy. Used for perf-only intermediate shapes where no golden reference exists.
async fn run_shape_synth(
    device: &wgpu::Device,
    queue:  &wgpu::Queue,
    bgl:    &wgpu::BindGroupLayout,
    pipe_qkt: &wgpu::ComputePipeline,
    pipe_sm:  &wgpu::ComputePipeline,
    pipe_pv:  &wgpu::ComputePipeline,
    shape:  &Shape,
    iters:  usize,
    has_ts: bool,
) -> f64 {
    println!("[{}] B={} Nq={} Nkv={} S={} D={} groups={}",
        shape.name, shape.batch, shape.n_q, shape.n_kv, shape.seq, shape.d_head, shape.groups());

    // Deterministic synthetic inputs (same pattern as wgpu-matmul — tiny values, bounded).
    let scale_in = 1.0_f32 / (shape.d_head as f32).sqrt();
    let q: Vec<f32> = (0..shape.q_elems()).map(|i| ((i % 7) as f32) * 0.01 * scale_in).collect();
    let k: Vec<f32> = (0..shape.kv_elems()).map(|i| ((i % 11) as f32) * 0.01 * scale_in).collect();
    let v: Vec<f32> = (0..shape.kv_elems()).map(|i| ((i % 13) as f32) * 0.01).collect();

    let params = Params {
        batch:  shape.batch, n_q: shape.n_q, n_kv: shape.n_kv,
        seq:    shape.seq,    d_head: shape.d_head, groups: shape.groups(),
        _pad0: 0, _pad1: 0, scale:  shape.scale(),
        _pad2: 0.0, _pad3: 0.0, _pad4: 0.0,
    };

    let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buf_q = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Q"), contents: bytemuck::cast_slice(&q), usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_k = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("K"), contents: bytemuck::cast_slice(&k), usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("V"), contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scores"), size: (shape.scores_elems() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Out"), size: (shape.q_elems() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn-bg"), layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buf_scores.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: buf_out.as_entire_binding() },
        ],
    });

    let ts_count: u32 = 6;
    let ts_set = if has_ts {
        Some(device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("ts"), ty: wgpu::QueryType::Timestamp, count: ts_count,
        }))
    } else { None };
    let ts_resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_resolve"), size: (ts_count as u64) * 8,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let ts_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_read"), size: (ts_count as u64) * 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let ts_period = if has_ts { queue.get_timestamp_period() as f64 } else { 0.0 };

    let wg_qkt = ((shape.seq + 15) / 16, (shape.seq + 15) / 16, shape.batch * shape.n_q);
    let wg_sm  = (shape.batch * shape.n_q * shape.seq, 1, 1);
    let wg_pv  = ((shape.d_head + 15) / 16, (shape.seq + 15) / 16, shape.batch * shape.n_q);

    let total_flops = shape.flops();
    println!("[{}] flops/iter = {:.2} GFLOP   scores buf = {:.1} MiB",
        shape.name, total_flops / 1e9, shape.scores_elems() as f64 * 4.0 / (1024.0*1024.0));

    let mut total_ms_vec: Vec<f64> = Vec::new();
    let mut qkt_ms_vec: Vec<f64> = Vec::new();
    let mut sm_ms_vec:  Vec<f64> = Vec::new();
    let mut pv_ms_vec:  Vec<f64> = Vec::new();

    for iter in 0..(iters + 1) {
        let is_warmup = iter == 0;
        let label = if is_warmup { "warmup" } else { "iter" };
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (pipe, begin, end, name) in [
            (pipe_qkt, 0u32, 1u32, "qkt"),
            (pipe_sm,  2u32, 3u32, "softmax"),
            (pipe_pv,  4u32, 5u32, "pv"),
        ] {
            let timestamp_writes = ts_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(begin),
                end_of_pass_write_index: Some(end),
            });
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(name), timestamp_writes });
            p.set_pipeline(pipe);
            p.set_bind_group(0, &bind_group, &[]);
            let wg = match name { "qkt" => wg_qkt, "softmax" => wg_sm, _ => wg_pv };
            p.dispatch_workgroups(wg.0, wg.1, wg.2);
        }
        if let Some(qs) = ts_set.as_ref() {
            enc.resolve_query_set(qs, 0..ts_count, &ts_resolve, 0);
            enc.copy_buffer_to_buffer(&ts_resolve, 0, &ts_read, 0, (ts_count as u64) * 8);
        }
        let cpu_start = Instant::now();
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        let (qkt_ms, sm_ms, pv_ms, total_ms) = if has_ts {
            let slice = ts_read.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let ts: &[u64] = bytemuck::cast_slice(&data);
            let ns = |a: u64, b: u64| (b.wrapping_sub(a)) as f64 * ts_period;
            let r = (ns(ts[0], ts[1])/1e6, ns(ts[2], ts[3])/1e6, ns(ts[4], ts[5])/1e6,
                     ns(ts[0], ts[5])/1e6);
            drop(data);
            ts_read.unmap();
            r
        } else {
            (f64::NAN, f64::NAN, f64::NAN, cpu_ms)
        };

        let tflops = (total_flops / 1e12) / (total_ms / 1000.0);
        println!("[{}] {} {}:  total={:.3} ms  qkt={:.3} sm={:.3} pv={:.3}  ({:.4} TFLOPS)  cpu_wall={:.2} ms",
            shape.name, label, iter, total_ms, qkt_ms, sm_ms, pv_ms, tflops, cpu_ms);
        if !is_warmup {
            total_ms_vec.push(total_ms);
            if !qkt_ms.is_nan() { qkt_ms_vec.push(qkt_ms); sm_ms_vec.push(sm_ms); pv_ms_vec.push(pv_ms); }
        }
    }

    total_ms_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = total_ms_vec[0];
    let med  = total_ms_vec[total_ms_vec.len()/2];
    let best_tf = (total_flops / 1e12) / (best / 1000.0);
    let med_tf  = (total_flops / 1e12) / (med  / 1000.0);
    println!("[{}] BEST   total={:.3} ms  {:.4} TFLOPS", shape.name, best, best_tf);
    println!("[{}] MEDIAN total={:.3} ms  {:.4} TFLOPS", shape.name, med, med_tf);
    if !qkt_ms_vec.is_empty() {
        qkt_ms_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sm_ms_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pv_ms_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let s = total_ms_vec.iter().sum::<f64>() / total_ms_vec.len() as f64;
        println!("[{}] per-stage median:  qkt={:.3} ms ({:.1}%)  sm={:.3} ms ({:.1}%)  pv={:.3} ms ({:.1}%)",
            shape.name,
            qkt_ms_vec[qkt_ms_vec.len()/2], qkt_ms_vec[qkt_ms_vec.len()/2] / s * 100.0,
            sm_ms_vec[sm_ms_vec.len()/2],   sm_ms_vec[sm_ms_vec.len()/2]   / s * 100.0,
            pv_ms_vec[pv_ms_vec.len()/2],   pv_ms_vec[pv_ms_vec.len()/2]   / s * 100.0,
        );
    }
    best_tf
}
