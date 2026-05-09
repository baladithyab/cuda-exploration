// Rudimentary 2D Gaussian Splatting forward rasterizer in cuda-oxide.
// Wave 8: proof-of-life test of cuda-oxide's expressiveness on a non-matmul
// non-reduction kernel with control flow, early-exit, and libdevice math.
//
// Wave 8.5: better scene generation. Two procedural scenes (rings, smiley)
// instead of random noise. Kernel is unchanged.
//
// Algorithm: per pixel, iterate over all N gaussians in pre-sorted order,
// front-to-back alpha blend. No tile binning, no SH, no 3D projection.
// See ANALYSIS.md for simplifications.

#![feature(core_intrinsics)]
#![allow(internal_features)]

use cuda_core::{CudaContext, CudaEvent, DeviceBuffer, LaunchConfig, sys};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

const W: u32 = 256;
const H: u32 = 256;
const BS: u32 = 16;

#[kernel]
pub fn rasterize_2dgs(
    means_x: &[f32],
    means_y: &[f32],
    conic_xx: &[f32],
    conic_xy: &[f32],
    conic_yy: &[f32],
    opacity: &[f32],
    color_r: &[f32],
    color_g: &[f32],
    color_b: &[f32],
    n_gaussians: u32,
    width: u32,
    height: u32,
    mut out_r: DisjointSlice<f32>,
    mut out_g: DisjointSlice<f32>,
    mut out_b: DisjointSlice<f32>,
) {
    let px = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    let py = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();
    if px >= width || py >= height {
        return;
    }
    let pxf = px as f32;
    let pyf = py as f32;
    let pidx = (py * width + px) as usize;
    let n = n_gaussians as usize;

    let mut accum_r: f32 = 0.0;
    let mut accum_g: f32 = 0.0;
    let mut accum_b: f32 = 0.0;
    let mut transmittance: f32 = 1.0;

    let mut i: usize = 0;
    while i < n {
        let dx = pxf - means_x[i];
        let dy = pyf - means_y[i];
        let power = -0.5
            * (conic_xx[i] * dx * dx + 2.0 * conic_xy[i] * dx * dy + conic_yy[i] * dy * dy);
        if power <= 0.0 {
            // exp(power) via libdevice __nv_expf (cuda-oxide lowers expf32 to this).
            let alpha = opacity[i] * unsafe { core::intrinsics::expf32(power) };
            if alpha >= 1.0 / 255.0 {
                let alpha_clamped = if alpha > 0.99 { 0.99 } else { alpha };
                let weight = alpha_clamped * transmittance;
                accum_r = accum_r + weight * color_r[i];
                accum_g = accum_g + weight * color_g[i];
                accum_b = accum_b + weight * color_b[i];
                transmittance = transmittance * (1.0 - alpha_clamped);
                if transmittance < 0.0001 {
                    unsafe {
                        *out_r.as_mut_ptr().add(pidx) = accum_r;
                        *out_g.as_mut_ptr().add(pidx) = accum_g;
                        *out_b.as_mut_ptr().add(pidx) = accum_b;
                    }
                    return;
                }
            }
        }
        i += 1;
    }

    unsafe {
        *out_r.as_mut_ptr().add(pidx) = accum_r;
        *out_g.as_mut_ptr().add(pidx) = accum_g;
        *out_b.as_mut_ptr().add(pidx) = accum_b;
    }
}

// ---------- Host side ----------

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f01(&mut self) -> f32 { (self.next_u32() as f32) / (u32::MAX as f32) }
    fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + self.next_f01() * (hi - lo) }
}

#[derive(Clone, Copy)]
struct Gaussian {
    mx: f32, my: f32,
    sx: f32, sy: f32,
    opacity: f32,
    r: f32, g: f32, b: f32,
    depth: f32,
}

// HSV->RGB with h in [0,1), s,v in [0,1].
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h6 = (h.rem_euclid(1.0)) * 6.0;
    let i = h6.floor() as i32;
    let f = h6 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

/// Scene A: concentric rainbow rings.
fn scene_rings() -> Vec<Gaussian> {
    let mut rng = Lcg::new(0xA1B2_C3D4_E5F6_0001);
    let mut v: Vec<Gaussian> = Vec::new();
    let cx = 128.0_f32;
    let cy = 128.0_f32;
    // 16 rings from r=8 to r=120, 256 gaussians per ring = 4096 total.
    let n_rings = 16;
    let per_ring = 256;
    for ring in 0..n_rings {
        let t = ring as f32 / (n_rings - 1) as f32;
        let radius = 8.0 + t * (120.0 - 8.0);
        // Rainbow hue by radius.
        let (r, g, b) = hsv_to_rgb(t * 0.9, 0.9, 1.0);
        for k in 0..per_ring {
            let base_ang = (k as f32 / per_ring as f32) * std::f32::consts::TAU;
            let jitter = rng.range(-0.5, 0.5) * (std::f32::consts::TAU / per_ring as f32);
            let ang = base_ang + jitter;
            let mx = cx + radius * ang.cos();
            let my = cy + radius * ang.sin();
            let sigma = rng.range(4.0, 6.0);
            v.push(Gaussian {
                mx, my,
                sx: sigma, sy: sigma,
                opacity: rng.range(0.7, 0.9),
                r, g, b,
                depth: -radius, // outer rings farther back; inner rings in front.
            });
        }
    }
    // Front-to-back blend: the FIRST gaussian is frontmost. "depth = -radius"
    // means smaller radius -> more-negative -> more-front. So we sort
    // ASCENDING (most negative first = innermost ring drawn first = on top).
    v.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
    v
}

/// Scene B: smiley face on sky-blue background.
fn scene_smiley() -> Vec<Gaussian> {
    let mut rng = Lcg::new(0xFACE_5A11_3B00_0002);
    let mut v: Vec<Gaussian> = Vec::new();

    // Background: 3000 gaussians on a jittered coarse grid, sky-blue.
    let sky = (135.0 / 255.0, 206.0 / 255.0, 235.0 / 255.0);
    let grid = 55; // 55*55 = 3025
    for iy in 0..grid {
        for ix in 0..grid {
            let sx_pos = (ix as f32 + 0.5) * (W as f32 / grid as f32) + rng.range(-2.0, 2.0);
            let sy_pos = (iy as f32 + 0.5) * (H as f32 / grid as f32) + rng.range(-2.0, 2.0);
            v.push(Gaussian {
                mx: sx_pos, my: sy_pos,
                sx: 12.0, sy: 12.0,
                opacity: 0.3,
                r: sky.0, g: sky.1, b: sky.2,
                depth: -1.0, // background: smallest magnitude -> rendered LAST in ascending sort.
            });
        }
    }

    // Face disk: 2000 yellow gaussians on a disk of radius 80 around center.
    let cx = 128.0_f32;
    let cy = 128.0_f32;
    let face_r_max = 80.0_f32;
    let n_face = 2000;
    for _ in 0..n_face {
        // Uniform on disk: r = R*sqrt(u), theta = 2*pi*v
        let u = rng.next_f01();
        let w = rng.next_f01();
        let r = face_r_max * u.sqrt();
        let ang = w * std::f32::consts::TAU;
        let mx = cx + r * ang.cos();
        let my = cy + r * ang.sin();
        v.push(Gaussian {
            mx, my,
            sx: 6.0, sy: 6.0,
            opacity: 0.85,
            r: 1.0, g: 0.85, b: 0.0,
            depth: -100.0,
        });
    }

    // Eyes: 250 black gaussians at each eye center.
    for eye_cx in [104.0_f32, 152.0_f32] {
        let eye_cy = 108.0_f32;
        for _ in 0..250 {
            // small disk of radius 6 around the eye center.
            let u = rng.next_f01();
            let w = rng.next_f01();
            let r = 6.0 * u.sqrt();
            let ang = w * std::f32::consts::TAU;
            v.push(Gaussian {
                mx: eye_cx + r * ang.cos(),
                my: eye_cy + r * ang.sin(),
                sx: 4.0, sy: 4.0,
                opacity: 0.95,
                r: 0.0, g: 0.0, b: 0.0,
                depth: -1000.0, // most-negative -> frontmost in ascending sort.
            });
        }
    }

    // Mouth: 500 black gaussians on a lower arc. Center of mouth circle = (128, 128),
    // radius 30, but only keep points with y > 140 (lower half, shifted).
    let mouth_cx = 128.0_f32;
    let mouth_cy = 128.0_f32;
    let mouth_r = 30.0_f32;
    let mut placed = 0;
    while placed < 500 {
        // Angle in lower half: pi/6 .. 5pi/6 (below center).
        let ang = rng.range(std::f32::consts::PI * 0.2, std::f32::consts::PI * 0.8);
        let my = mouth_cy + mouth_r * ang.sin();
        if my <= 140.0 { continue; }
        let mx = mouth_cx + mouth_r * ang.cos();
        v.push(Gaussian {
            mx, my,
            sx: 3.0, sy: 3.0,
            opacity: 0.9,
            r: 0.0, g: 0.0, b: 0.0,
            depth: -1000.0,
        });
        placed += 1;
    }

    // Ascending sort: most-negative depth first -> frontmost first.
    // eyes/mouth (-1000) -> face (-100) -> background (-1). Frontmost drawn first. ✓
    v.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
    v
}

fn save_ppm(path: &str, pixels_r: &[f32], pixels_g: &[f32], pixels_b: &[f32], w: u32, h: u32) {
    let f = File::create(path).expect("create ppm");
    let mut bw = BufWriter::new(f);
    writeln!(bw, "P6").unwrap();
    writeln!(bw, "{} {}", w, h).unwrap();
    writeln!(bw, "255").unwrap();
    let n = (w * h) as usize;
    let mut buf: Vec<u8> = Vec::with_capacity(n * 3);
    for i in 0..n {
        let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        buf.push(to_u8(pixels_r[i]));
        buf.push(to_u8(pixels_g[i]));
        buf.push(to_u8(pixels_b[i]));
    }
    bw.write_all(&buf).unwrap();
}

fn render_scene(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &std::sync::Arc<cuda_core::CudaModule>,
    gs: &[Gaussian],
    label: &str,
    out_ppm: &str,
    iters: usize,
) {
    let n = gs.len();
    let mut mx = Vec::with_capacity(n);
    let mut my = Vec::with_capacity(n);
    let mut cxx = Vec::with_capacity(n);
    let mut cxy = Vec::with_capacity(n);
    let mut cyy = Vec::with_capacity(n);
    let mut op  = Vec::with_capacity(n);
    let mut cr  = Vec::with_capacity(n);
    let mut cg  = Vec::with_capacity(n);
    let mut cb  = Vec::with_capacity(n);
    for g in gs {
        let a = g.sx * g.sx;
        let c = g.sy * g.sy;
        let b = 0.0_f32;
        let det = a * c - b * b;
        mx.push(g.mx); my.push(g.my);
        cxx.push(c / det);
        cxy.push(-b / det);
        cyy.push(a / det);
        op.push(g.opacity);
        cr.push(g.r); cg.push(g.g); cb.push(g.b);
    }

    let d_mx  = DeviceBuffer::from_host(stream, &mx).unwrap();
    let d_my  = DeviceBuffer::from_host(stream, &my).unwrap();
    let d_cxx = DeviceBuffer::from_host(stream, &cxx).unwrap();
    let d_cxy = DeviceBuffer::from_host(stream, &cxy).unwrap();
    let d_cyy = DeviceBuffer::from_host(stream, &cyy).unwrap();
    let d_op  = DeviceBuffer::from_host(stream, &op).unwrap();
    let d_cr  = DeviceBuffer::from_host(stream, &cr).unwrap();
    let d_cg  = DeviceBuffer::from_host(stream, &cg).unwrap();
    let d_cb  = DeviceBuffer::from_host(stream, &cb).unwrap();

    let pixels = (W * H) as usize;
    let mut d_out_r = DeviceBuffer::<f32>::zeroed(stream, pixels).unwrap();
    let mut d_out_g = DeviceBuffer::<f32>::zeroed(stream, pixels).unwrap();
    let mut d_out_b = DeviceBuffer::<f32>::zeroed(stream, pixels).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (W.div_ceil(BS), H.div_ceil(BS), 1),
        block_dim: (BS, BS, 1),
        shared_mem_bytes: 0,
    };
    let n_arg: u32 = n as u32;

    // Warmup.
    {
        let s = stream.clone();
        let m = module.clone();
        cuda_launch! {
            kernel: rasterize_2dgs, stream: s, module: m, config: cfg,
            args: [slice(&d_mx), slice(&d_my),
                   slice(&d_cxx), slice(&d_cxy), slice(&d_cyy),
                   slice(&d_op),
                   slice(&d_cr), slice(&d_cg), slice(&d_cb),
                   n_arg, W, H,
                   slice_mut(&mut d_out_r), slice_mut(&mut d_out_g), slice_mut(&mut d_out_b)]
        }.unwrap();
        stream.synchronize().unwrap();
    }

    let mut times_ms = Vec::<f64>::new();
    for i in 0..iters {
        let start: CudaEvent = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
        let stop:  CudaEvent = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
        let s = stream.clone();
        let m = module.clone();
        let t0 = Instant::now();
        start.record(stream).unwrap();
        cuda_launch! {
            kernel: rasterize_2dgs, stream: s, module: m, config: cfg,
            args: [slice(&d_mx), slice(&d_my),
                   slice(&d_cxx), slice(&d_cxy), slice(&d_cyy),
                   slice(&d_op),
                   slice(&d_cr), slice(&d_cg), slice(&d_cb),
                   n_arg, W, H,
                   slice_mut(&mut d_out_r), slice_mut(&mut d_out_g), slice_mut(&mut d_out_b)]
        }.unwrap();
        stop.record(stream).unwrap();
        stream.synchronize().unwrap();
        let gpu_ms = start.elapsed_ms(&stop).unwrap() as f64;
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("[{label}] iter {i}: gpu_ms={gpu_ms:.3} cpu_wall_ms={cpu_ms:.3}");
        times_ms.push(gpu_ms);
    }

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times_ms[times_ms.len() / 2];
    let best = times_ms[0];
    println!("[{label}] N={n} median_ms={med:.3}  best_ms={best:.3}");

    let h_r = d_out_r.to_host_vec(stream).unwrap();
    let h_g = d_out_g.to_host_vec(stream).unwrap();
    let h_b = d_out_b.to_host_vec(stream).unwrap();
    save_ppm(out_ppm, &h_r, &h_g, &h_b, W, H);
    println!("[{label}] wrote {out_ppm}");
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_3dgs_mini").expect("load module");

    let manifest = env!("CARGO_MANIFEST_DIR");

    // Scene A: rings.
    let gs_rings = scene_rings();
    let path_rings = format!("{}/output_rings.ppm", manifest);
    render_scene(&ctx, &stream, &module, &gs_rings, "rings", &path_rings, 3);

    // Scene B: smiley.
    let gs_smiley = scene_smiley();
    let path_smiley = format!("{}/output_smiley.ppm", manifest);
    render_scene(&ctx, &stream, &module, &gs_smiley, "smiley", &path_smiley, 3);

    // Keep a default output.ppm pointing at the rings scene for backwards-compat.
    let path_default = format!("{}/output.ppm", manifest);
    std::fs::copy(&path_rings, &path_default).ok();
}
