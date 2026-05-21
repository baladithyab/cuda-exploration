// Wave C1.6 — wgpu/WGSL naive 3-pass Multi-Head Latent Attention (MLA).
//
// Cross-frontend × cross-mechanism cell (row=MLA, col=wgpu).
// Algorithm reference: cuda-attn-mla/attn_mla.cu (3-kernel decomposition).
// Stack reference:     wgpu-attn-gqa/         (same wgpu/WGSL plumbing).
//
// Structural difference vs GQA:
//   - MLA has a single shared head index for Q,K,V (no n_kv != n_q broadcast).
//   - Per-head dims are non-square: qk_head_dim (192) != d_v (128).
// Both are reflected in the WGSL Params layout and indexers.
//
// **WebGPU has NO tensor cores** — kernels run pure FFMA in WGSL. Expected
// perf: in the FFMA-no-TC ballpark of oxide-attn-mla (24.70 TF best at the
// deepseek_v3 shape on RTX 5090). Realistically wgpu adds Vulkan-stack
// overhead so the target is the 5–15 TF range on a real GPU.
//
// **WSL2 dead-end:** the only adapters wgpu sees here are llvmpipe CPU
// rasterizers; correctness numbers below are real, timing numbers from
// llvmpipe are NOT GPU perf. Same caveat as wgpu-attn-gqa wave 16.2.
//
// **No FP16 in WGSL by default:** `wgpu::Features::SHADER_F16` is not
// available on most llvmpipe / WSL paths. We load the *_f32.npy variants
// of the MLA inputs (numerically identical to f16 inputs cast to f32 by
// the reference generator), avoiding any host-side bf16/f16→f32 cast.
//
// Shapes (canonical from analysis/wave15-attention-architecture/inputs/):
//   correctness_mla : B=1 n_h=4   S=128  qk=96  d_v=64
//   deepseek_v3     : B=1 n_h=128 S=2048 qk=192 d_v=128  (scores buf = 2 GiB)
//
// The deepseek_v3 scores buffer (1·128·2048·2048·4 = 2.0 GiB) exceeds even
// most real-GPU max_storage_buffer_binding_size caps (often 2 GiB minus
// alignment slack) and *all* llvmpipe caps (128 MiB). It will SKIP under
// llvmpipe; on a real GPU that supports it, the bench shape will run.
//
// We also include a scaled-down `medium_mla` perf-sanity shape (synthetic
// inputs) to give an actual TFLOPS readout under llvmpipe even when the
// canonical bench is skipped — same pattern as wgpu-attn-gqa's medium_512.
//
// FLOPs/iter (per analysis/.../reference/flops_mla.py):
//   2 · B · n_h · S² · (qk + d_v)

use std::path::PathBuf;
use std::time::Instant;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Copy)]
struct Shape {
    name: &'static str,
    batch: u32,
    n_h:   u32,
    seq:   u32,
    qk:    u32,
    d_v:   u32,
}

impl Shape {
    fn scale(&self) -> f32 { 1.0_f32 / (self.qk as f32).sqrt() }
    fn q_elems(&self) -> usize {
        (self.batch * self.n_h * self.seq * self.qk) as usize
    }
    fn v_elems(&self) -> usize {
        (self.batch * self.n_h * self.seq * self.d_v) as usize
    }
    fn scores_elems(&self) -> usize {
        (self.batch * self.n_h * self.seq * self.seq) as usize
    }
    fn flops(&self) -> f64 {
        let (b, h, s, qk, dv) =
            (self.batch as f64, self.n_h as f64, self.seq as f64,
             self.qk as f64, self.d_v as f64);
        2.0 * b * h * s * s * (qk + dv)
    }
    /// per Wave C1.6 acceptance: max_abs_err <= 1e-2 sampled vs PyTorch SDPA-MLA.
    fn atol(&self) -> f32 { 1e-2 }
    fn rtol(&self) -> f32 { 1e-3 }
}

const CORRECTNESS:  Shape = Shape { name: "correctness_mla", batch: 1, n_h: 4,   seq: 128,  qk: 96,  d_v: 64  };
// Synthetic mid shape: small enough for llvmpipe (scores ≈ 8 MiB) yet large
// enough that the FFMA inner loop dominates timing. n_h=4, S=512, qk=96, d_v=64.
const MEDIUM_MLA:   Shape = Shape { name: "medium_mla",      batch: 1, n_h: 4,   seq: 512,  qk: 96,  d_v: 64  };
const DEEPSEEK_V3:  Shape = Shape { name: "deepseek_v3",     batch: 1, n_h: 128, seq: 2048, qk: 192, d_v: 128 };

fn load_npy_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {:?}: {e}", path));
    let reader = npyz::NpyFile::new(&bytes[..]).expect("parse npy header");
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    batch:  u32,
    n_h:    u32,
    seq:    u32,
    qk:     u32,
    d_v:    u32,
    _pad0:  u32,
    _pad1:  u32,
    _pad2:  u32,
    scale:  f32,
    _pad3:  f32,
    _pad4:  f32,
    _pad5:  f32,
}

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
        println!("[wgpu] candidate: {:<50} backend={:?}  type={:?}",
            info.name, info.backend, info.device_type);
        if info.device_type != wgpu::DeviceType::Cpu && chosen.is_none() {
            chosen = Some(ad);
        } else if fallback.is_none() {
            fallback = Some(ad);
        }
    }
    let adapter = chosen.or(fallback).expect("no adapter");
    let info = adapter.get_info();
    let is_cpu = info.device_type == wgpu::DeviceType::Cpu;
    println!("\n[wgpu] using: {} ({:?}, type={:?})",
        info.name, info.backend, info.device_type);
    if is_cpu {
        println!("[wgpu] !! WARNING: only CPU adapters enumerated. WSL2 dead-end.");
        println!("[wgpu] !! Kernel runs via llvmpipe on CPU — TFLOPS below is NOT GPU perf.");
        println!("[wgpu] !! Correctness IS meaningful; timing is not.");
    } else {
        println!("[wgpu] GPU adapter — numbers below reflect real hardware.");
    }

    let features_avail = adapter.features();
    let want_ts = wgpu::Features::TIMESTAMP_QUERY;
    let req_features = if features_avail.contains(want_ts) { want_ts } else { wgpu::Features::empty() };

    let adapter_limits = adapter.limits();
    let mut limits = adapter_limits.clone();
    limits.max_compute_invocations_per_workgroup =
        limits.max_compute_invocations_per_workgroup.max(256);

    println!("[wgpu] adapter caps: max_storage_buffer_binding_size = {} MiB, max_buffer_size = {} MiB",
        adapter_limits.max_storage_buffer_binding_size / (1024 * 1024),
        adapter_limits.max_buffer_size / (1024 * 1024));

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu-attn-mla"),
            required_features: req_features,
            required_limits: limits,
            ..Default::default()
        }, None)
        .await
        .expect("device");
    let has_ts = req_features.contains(wgpu::Features::TIMESTAMP_QUERY);
    println!("[wgpu] timestamp_query feature: {has_ts}\n");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("attn-mla.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("attn.wgsl").into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("attn-mla-bgl"),
        entries: &[
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
            bgl_storage_entry(1, true),
            bgl_storage_entry(2, true),
            bgl_storage_entry(3, true),
            bgl_storage_entry(4, false),
            bgl_storage_entry(5, false),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("attn-mla-pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let make_pipe = |entry: &'static str| -> wgpu::ComputePipeline {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: entry,
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pipe_qkt = make_pipe("mla_qkt");
    let pipe_sm  = make_pipe("softmax");
    let pipe_pv  = make_pipe("mla_pv");

    let inputs_dir: PathBuf = PathBuf::from(
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs",
    );
    println!("[wgpu] inputs dir = {:?}\n", inputs_dir);

    println!("=== SHAPE: correctness_mla ===");
    let correctness_ok = run_shape(
        &device, &queue, &bgl, &pipe_qkt, &pipe_sm, &pipe_pv,
        &CORRECTNESS, &inputs_dir, /*iters=*/ 2, has_ts,
    ).await;

    if !correctness_ok {
        println!("\n[wgpu] correctness FAILED — skipping bench shape.");
        return;
    }

    println!("\n=== SHAPE: medium_mla (perf-only, synthetic inputs) ===");
    println!("[medium_mla] note: deepseek_v3 needs 2 GiB scores (n_h=128, S=2048).");
    println!("[medium_mla]       medium_mla is a scaled-down perf sanity (n_h=4 S=512).");
    let medium_need = (MEDIUM_MLA.scores_elems() as u64) * 4;
    if medium_need > device.limits().max_storage_buffer_binding_size as u64 {
        println!("[medium_mla] SKIP: scores {} MiB > max {} MiB",
            medium_need / (1024*1024),
            device.limits().max_storage_buffer_binding_size / (1024*1024));
    } else {
        let _ = run_shape_synth(
            &device, &queue, &bgl, &pipe_qkt, &pipe_sm, &pipe_pv,
            &MEDIUM_MLA, /*iters=*/ 3, has_ts,
        ).await;
    }

    println!("\n=== SHAPE: deepseek_v3 (canonical bench) ===");
    let need = (DEEPSEEK_V3.scores_elems() as u64) * 4;
    if need > device.limits().max_storage_buffer_binding_size as u64 {
        println!("[deepseek_v3] SKIP: scores buffer {} MiB exceeds max_storage_buffer_binding_size {} MiB",
            need / (1024*1024),
            device.limits().max_storage_buffer_binding_size / (1024*1024));
        println!("[deepseek_v3] scores = B·n_h·S² · 4B = 1·128·2048²·4 = 2 GiB.");
        println!("[deepseek_v3] llvmpipe caps at 128 MiB. To run, need a real GPU adapter or");
        println!("[deepseek_v3] a fused/FlashAttention-style scores-tile streaming rewrite.");
    } else {
        let _ = run_shape(
            &device, &queue, &bgl, &pipe_qkt, &pipe_sm, &pipe_pv,
            &DEEPSEEK_V3, &inputs_dir, /*iters=*/ 4, has_ts,
        ).await;
    }

    if is_cpu {
        println!("\n[wgpu] Reminder: these times are llvmpipe on CPU. NOT GPU perf.");
        println!("[wgpu] Wave C1.6 baselines for cross-API context (real GPU, RTX 5090):");
        println!("[wgpu]   cuda-attn-mla:  24.17 TF (HMMA)");
        println!("[wgpu]   oxide-attn-mla: 24.70 TF (FFMA, no TC) -- closest peer");
        println!("[wgpu] WGSL has no tensor cores; expected real-GPU perf class is FFMA.");
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
    println!("[{}] B={} n_h={} S={} qk={} d_v={}",
        shape.name, shape.batch, shape.n_h, shape.seq, shape.qk, shape.d_v);

    let q = load_npy_f32(&inputs_dir.join(format!("mla_{}_q_f32.npy", shape.name)));
    let k = load_npy_f32(&inputs_dir.join(format!("mla_{}_k_f32.npy", shape.name)));
    let v = load_npy_f32(&inputs_dir.join(format!("mla_{}_v_f32.npy", shape.name)));
    let expected = load_npy_f32(&inputs_dir.join(format!("mla_{}_expected_f32.npy", shape.name)));
    assert_eq!(q.len(), shape.q_elems(), "Q elem mismatch");
    assert_eq!(k.len(), shape.q_elems(), "K elem mismatch (uses qk dim)");
    assert_eq!(v.len(), shape.v_elems(), "V elem mismatch (uses d_v dim)");
    assert_eq!(expected.len(), shape.v_elems(), "expected has d_v dim");
    println!("[{}] loaded Q+K ({:.1} MiB ea)  V ({:.1} MiB)  expected ({:.1} MiB)",
        shape.name,
        q.len() as f64 * 4.0 / (1024.0 * 1024.0),
        v.len() as f64 * 4.0 / (1024.0 * 1024.0),
        expected.len() as f64 * 4.0 / (1024.0 * 1024.0),
    );

    let params = Params {
        batch:  shape.batch,
        n_h:    shape.n_h,
        seq:    shape.seq,
        qk:     shape.qk,
        d_v:    shape.d_v,
        _pad0: 0, _pad1: 0, _pad2: 0,
        scale:  shape.scale(),
        _pad3: 0.0, _pad4: 0.0, _pad5: 0.0,
    };

    let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"), contents: bytemuck::bytes_of(&params),
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
    let out_bytes    = (shape.v_elems() * 4) as u64;
    let buf_scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scores"), size: scores_bytes,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Out"), size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"), size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn-mla-bg"), layout: bgl,
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

    // Dispatch geometry. WGSL workgroup_size is (16,16,1) for QKt and PV; one
    // thread per (j,i) in QKt and one per (d,i) in PV. n_h replaces n_q in z.
    let wg_qkt = ((shape.seq + 15) / 16, (shape.seq + 15) / 16, shape.batch * shape.n_h);
    let wg_sm  = (shape.batch * shape.n_h * shape.seq, 1, 1);
    let wg_pv  = ((shape.d_v + 15) / 16, (shape.seq + 15) / 16, shape.batch * shape.n_h);

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
        if iter == iters {
            enc.copy_buffer_to_buffer(&buf_out, 0, &buf_readback, 0, out_bytes);
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

    total_ms_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = total_ms_vec.first().copied().unwrap_or(f64::NAN);
    let med  = total_ms_vec.get(total_ms_vec.len()/2).copied().unwrap_or(f64::NAN);
    let best_tf = (total_flops / 1e12) / (best / 1000.0);
    let med_tf  = (total_flops / 1e12) / (med  / 1000.0);
    println!("[{}] BEST   total={:.3} ms  {:.4} TFLOPS", shape.name, best, best_tf);
    println!("[{}] MEDIAN total={:.3} ms  {:.4} TFLOPS", shape.name, med,  med_tf);
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
    println!("[{}] B={} n_h={} S={} qk={} d_v={}",
        shape.name, shape.batch, shape.n_h, shape.seq, shape.qk, shape.d_v);

    let scale_in = 1.0_f32 / (shape.qk as f32).sqrt();
    let q: Vec<f32> = (0..shape.q_elems()).map(|i| ((i % 7) as f32) * 0.01 * scale_in).collect();
    let k: Vec<f32> = (0..shape.q_elems()).map(|i| ((i % 11) as f32) * 0.01 * scale_in).collect();
    let v: Vec<f32> = (0..shape.v_elems()).map(|i| ((i % 13) as f32) * 0.01).collect();

    let params = Params {
        batch: shape.batch, n_h: shape.n_h, seq: shape.seq,
        qk: shape.qk, d_v: shape.d_v,
        _pad0:0, _pad1:0, _pad2:0,
        scale: shape.scale(),
        _pad3:0.0, _pad4:0.0, _pad5:0.0,
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
        label: Some("Out"), size: (shape.v_elems() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn-mla-bg-synth"), layout: bgl,
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

    let wg_qkt = ((shape.seq + 15) / 16, (shape.seq + 15) / 16, shape.batch * shape.n_h);
    let wg_sm  = (shape.batch * shape.n_h * shape.seq, 1, 1);
    let wg_pv  = ((shape.d_v + 15) / 16, (shape.seq + 15) / 16, shape.batch * shape.n_h);

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
    let med_tf  = (total_flops / 1e12) / (med / 1000.0);
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
