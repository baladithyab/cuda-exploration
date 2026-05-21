// Wave 11: CUDA C++ apples-to-apples reference for oxide-3dgs-real.
// Line-by-line port of oxide-3dgs-real/src/main.rs:
//   PLY parser -> per-gaussian 3D->2D projection (quat->R, scales->Sigma_3d,
//   perspective Jacobian -> Sigma_2d, conic invert, SH DC->RGB, sigmoid
//   opacity, depth sort) -> device rasterize_2dgs kernel (byte-equivalent
//   to the cuda-oxide kernel) -> PPM out.
// Renders all four cameras (A, B, C, D) to enable per-camera pixel-diff
// against oxide-3dgs-real/output_utsuho_plush_{A,C,D}.ppm.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

static constexpr int W = 800;
static constexpr int H = 800;
static constexpr int BS = 16;
static constexpr int ITERS = 3;

// ---------- Kernel (byte-equivalent to oxide-3dgs-mini rasterize_2dgs) ----------

__global__ void rasterize_2dgs(
    const float* __restrict__ means_x,
    const float* __restrict__ means_y,
    const float* __restrict__ conic_xx,
    const float* __restrict__ conic_xy,
    const float* __restrict__ conic_yy,
    const float* __restrict__ opacity,
    const float* __restrict__ color_r,
    const float* __restrict__ color_g,
    const float* __restrict__ color_b,
    int n_gaussians,
    int width,
    int height,
    float* __restrict__ out_r,
    float* __restrict__ out_g,
    float* __restrict__ out_b)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float pxf = (float)px;
    float pyf = (float)py;
    int pidx = py * width + px;

    float accum_r = 0.0f, accum_g = 0.0f, accum_b = 0.0f;
    float transmittance = 1.0f;

    for (int i = 0; i < n_gaussians; ++i) {
        float dx = pxf - means_x[i];
        float dy = pyf - means_y[i];
        float power = -0.5f * (conic_xx[i] * dx * dx
                             + 2.0f * conic_xy[i] * dx * dy
                             + conic_yy[i] * dy * dy);
        if (power <= 0.0f) {
            float alpha = opacity[i] * expf(power);
            if (alpha >= 1.0f / 255.0f) {
                float alpha_clamped = (alpha > 0.99f) ? 0.99f : alpha;
                float weight = alpha_clamped * transmittance;
                accum_r = accum_r + weight * color_r[i];
                accum_g = accum_g + weight * color_g[i];
                accum_b = accum_b + weight * color_b[i];
                transmittance = transmittance * (1.0f - alpha_clamped);
                if (transmittance < 0.0001f) {
                    out_r[pidx] = accum_r;
                    out_g[pidx] = accum_g;
                    out_b[pidx] = accum_b;
                    return;
                }
            }
        }
    }

    out_r[pidx] = accum_r;
    out_g[pidx] = accum_g;
    out_b[pidx] = accum_b;
}

// ---------- Error check helper ----------
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

// ---------- Raw gaussian + PLY parser ----------

struct RawGaussian {
    float x, y, z;
    float f_dc[3];
    // SH degree 3 "rest" coefficients (45 floats), per-channel layout:
    //   f_rest[0..15]  = R bands 1..3
    //   f_rest[15..30] = G bands 1..3
    //   f_rest[30..45] = B bands 1..3
    // Empty if PLY has no f_rest_* properties (= SH degree 0 only).
    std::vector<float> f_rest;
    float opacity_logit;
    float scale[3];
    float rot[4]; // w, x, y, z
};

static std::vector<RawGaussian> parse_ply(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "open ply: %s\n", path.c_str()); exit(1); }
    std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    // Locate "end_header\n"
    const char* needle = "end_header\n";
    size_t nlen = strlen(needle);
    size_t header_end = 0;
    for (size_t i = 0; i + nlen <= buf.size(); ++i) {
        if (memcmp(&buf[i], needle, nlen) == 0) { header_end = i + nlen; break; }
    }
    if (header_end == 0) { fprintf(stderr, "no end_header\n"); exit(1); }

    std::string header(&buf[0], header_end);
    size_t n_vertex = 0;
    std::vector<std::string> props;
    {
        size_t pos = 0;
        while (pos < header.size()) {
            size_t eol = header.find('\n', pos);
            if (eol == std::string::npos) eol = header.size();
            std::string line = header.substr(pos, eol - pos);
            pos = eol + 1;
            const char* vp = "element vertex ";
            const char* pp = "property float ";
            if (line.compare(0, strlen(vp), vp) == 0) {
                n_vertex = (size_t)atoll(line.c_str() + strlen(vp));
            } else if (line.compare(0, strlen(pp), pp) == 0) {
                std::string name = line.substr(strlen(pp));
                while (!name.empty() && (name.back() == ' ' || name.back() == '\r' || name.back() == '\t'))
                    name.pop_back();
                props.push_back(name);
            }
        }
    }
    size_t nprops = props.size();
    printf("PLY header: %zu vertices, %zu float props\n", n_vertex, nprops);

    auto idx = [&](const char* name) -> int {
        for (size_t k = 0; k < props.size(); ++k) if (props[k] == name) return (int)k;
        fprintf(stderr, "property '%s' not found\n", name); exit(1);
    };
    int ix = idx("x"), iy = idx("y"), iz = idx("z");
    int ifdc0 = idx("f_dc_0"), ifdc1 = idx("f_dc_1"), ifdc2 = idx("f_dc_2");
    int iop = idx("opacity");
    int is0 = idx("scale_0"), is1 = idx("scale_1"), is2 = idx("scale_2");
    int ir0 = idx("rot_0"), ir1 = idx("rot_1"), ir2 = idx("rot_2"), ir3 = idx("rot_3");

    // SH degree 3: 45 "rest" coefficients (f_rest_0..44). Optional.
    auto idx_opt = [&](const char* name) -> int {
        for (size_t k = 0; k < props.size(); ++k) if (props[k] == name) return (int)k;
        return -1;
    };
    int frest_idx[45];
    bool have_rest = true;
    for (int k = 0; k < 45; ++k) {
        char nm[32]; snprintf(nm, sizeof(nm), "f_rest_%d", k);
        int p = idx_opt(nm);
        if (p < 0) { have_rest = false; break; }
        frest_idx[k] = p;
    }
    printf("SH support: %s\n", have_rest ? "degree 3 (16 coefs/channel)" : "degree 0 only (DC)");

    const char* body = &buf[header_end];
    size_t body_len = buf.size() - header_end;
    size_t expected = n_vertex * nprops * 4;
    if (body_len != expected) {
        fprintf(stderr, "body size mismatch: got %zu expected %zu\n", body_len, expected);
        exit(1);
    }

    std::vector<RawGaussian> out;
    out.reserve(n_vertex);
    auto rf = [&](size_t base, int pi) -> float {
        size_t off = base + (size_t)pi * 4;
        float v; memcpy(&v, body + off, 4); return v;
    };
    for (size_t i = 0; i < n_vertex; ++i) {
        size_t base = i * nprops * 4;
        RawGaussian g;
        g.x = rf(base, ix); g.y = rf(base, iy); g.z = rf(base, iz);
        g.f_dc[0] = rf(base, ifdc0); g.f_dc[1] = rf(base, ifdc1); g.f_dc[2] = rf(base, ifdc2);
        g.opacity_logit = rf(base, iop);
        g.scale[0] = rf(base, is0); g.scale[1] = rf(base, is1); g.scale[2] = rf(base, is2);
        g.rot[0] = rf(base, ir0); g.rot[1] = rf(base, ir1); g.rot[2] = rf(base, ir2); g.rot[3] = rf(base, ir3);
        if (have_rest) {
            g.f_rest.resize(45);
            for (int k = 0; k < 45; ++k) g.f_rest[k] = rf(base, frest_idx[k]);
        }
        out.push_back(std::move(g));
    }
    return out;
}

// ---------- SH evaluation (degree 0 or 3) ----------
// Numerically identical to the Rust path in oxide-3dgs-real/src/main.rs.

static constexpr float SH_C0 = 0.28209479177387814f;
static constexpr float SH_C1 = 0.4886025119029199f;
static constexpr float SH_C2_0 =  1.0925484305920792f;
static constexpr float SH_C2_1 = -1.0925484305920792f;
static constexpr float SH_C2_2 =  0.31539156525252005f;
static constexpr float SH_C2_3 = -1.0925484305920792f;
static constexpr float SH_C2_4 =  0.5462742152960396f;
static constexpr float SH_C3_0 = -0.5900435899266435f;
static constexpr float SH_C3_1 =  2.890611442640554f;
static constexpr float SH_C3_2 = -0.4570457994644658f;
static constexpr float SH_C3_3 =  0.3731763325901154f;
static constexpr float SH_C3_4 = -0.4570457994644658f;
static constexpr float SH_C3_5 =  1.445305721320277f;
static constexpr float SH_C3_6 = -0.5900435899266435f;

static inline float sh_eval_one_channel(float dc, const float* rest15,
                                        float x, float y, float z) {
    // Band 0
    float r = SH_C0 * dc;
    // Band 1: -y, z, -x
    r = r + SH_C1 * (-y * rest15[0] + z * rest15[1] - x * rest15[2]);
    // Band 2
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, yz = y*z, xz = x*z;
    r = r + SH_C2_0 * xy * rest15[3]
          + SH_C2_1 * yz * rest15[4]
          + SH_C2_2 * (2.0f * zz - xx - yy) * rest15[5]
          + SH_C2_3 * xz * rest15[6]
          + SH_C2_4 * (xx - yy) * rest15[7];
    // Band 3
    r = r + SH_C3_0 * y * (3.0f * xx - yy) * rest15[8]
          + SH_C3_1 * xy * z * rest15[9]
          + SH_C3_2 * y * (4.0f * zz - xx - yy) * rest15[10]
          + SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * rest15[11]
          + SH_C3_4 * x * (4.0f * zz - xx - yy) * rest15[12]
          + SH_C3_5 * z * (xx - yy) * rest15[13]
          + SH_C3_6 * x * (xx - 3.0f * yy) * rest15[14];
    return r;
}

static inline void sh_to_rgb(const float f_dc[3], const std::vector<float>& f_rest,
                             float vx, float vy, float vz,
                             float* out_r, float* out_g, float* out_b) {
    if (f_rest.empty()) {
        *out_r = SH_C0 * f_dc[0] + 0.5f;
        *out_g = SH_C0 * f_dc[1] + 0.5f;
        *out_b = SH_C0 * f_dc[2] + 0.5f;
        return;
    }
    *out_r = sh_eval_one_channel(f_dc[0], f_rest.data() +  0, vx, vy, vz) + 0.5f;
    *out_g = sh_eval_one_channel(f_dc[1], f_rest.data() + 15, vx, vy, vz) + 0.5f;
    *out_b = sh_eval_one_channel(f_dc[2], f_rest.data() + 30, vx, vy, vz) + 0.5f;
}

// ---------- Host-side projection ----------

struct ProjGaussian {
    float mx, my;
    float cxx, cxy, cyy;
    float opacity;
    float r, g, b;
    float depth;
};

struct CamPose {
    float w[9]; // 3x3 row-major W
    float t[3];
    float fx, fy, cx, cy;
};

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

static void quat_to_mat3(const float q[4], float R[9]) {
    float n = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n < 1e-8f) n = 1e-8f;
    float w = q[0]/n, x = q[1]/n, y = q[2]/n, z = q[3]/n;
    R[0] = 1.0f - 2.0f*(y*y + z*z); R[1] = 2.0f*(x*y - w*z);       R[2] = 2.0f*(x*z + w*y);
    R[3] = 2.0f*(x*y + w*z);       R[4] = 1.0f - 2.0f*(x*x + z*z); R[5] = 2.0f*(y*z - w*x);
    R[6] = 2.0f*(x*z - w*y);       R[7] = 2.0f*(y*z + w*x);       R[8] = 1.0f - 2.0f*(x*x + y*y);
}
static void mat3_mul(const float a[9], const float b[9], float r[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k) s += a[i*3+k] * b[k*3+j];
            r[i*3+j] = s;
        }
}
static void mat3_transpose(const float a[9], float r[9]) {
    r[0]=a[0]; r[1]=a[3]; r[2]=a[6];
    r[3]=a[1]; r[4]=a[4]; r[5]=a[7];
    r[6]=a[2]; r[7]=a[5]; r[8]=a[8];
}

static void cov3d(const float rot[4], const float scale[3], float R[9]) {
    float Rm[9]; quat_to_mat3(rot, Rm);
    float Rt[9]; mat3_transpose(Rm, Rt);
    float sx = expf(scale[0]), sy = expf(scale[1]), sz = expf(scale[2]);
    float s2[9] = {sx*sx, 0.f, 0.f, 0.f, sy*sy, 0.f, 0.f, 0.f, sz*sz};
    float rs2[9]; mat3_mul(Rm, s2, rs2);
    mat3_mul(rs2, Rt, R);
}

struct ProjStats {
    size_t n_total, n_projected;
    size_t n_culled_behind, n_culled_far, n_culled_bad_cov;
    float depth_min, depth_max;
    float mean2d_min_x, mean2d_min_y, mean2d_max_x, mean2d_max_y;
    float conic_scale_median;
};

static std::vector<ProjGaussian> project_all(const std::vector<RawGaussian>& raws,
                                              const CamPose& cam,
                                              ProjStats* stats_out) {
    std::vector<ProjGaussian> out;
    out.reserve(raws.size());
    size_t n_culled_behind=0, n_culled_far=0, n_culled_bad_cov=0;
    float depth_min=INFINITY, depth_max=-INFINITY;
    float m2d_min_x=INFINITY, m2d_min_y=INFINITY;
    float m2d_max_x=-INFINITY, m2d_max_y=-INFINITY;
    std::vector<float> conic_scales;

    // Camera origin in world space: o = -W^T * t.
    float Wt_pose[9]; mat3_transpose(cam.w, Wt_pose);
    float cam_origin[3] = {
        -(Wt_pose[0]*cam.t[0] + Wt_pose[1]*cam.t[1] + Wt_pose[2]*cam.t[2]),
        -(Wt_pose[3]*cam.t[0] + Wt_pose[4]*cam.t[1] + Wt_pose[5]*cam.t[2]),
        -(Wt_pose[6]*cam.t[0] + Wt_pose[7]*cam.t[1] + Wt_pose[8]*cam.t[2]),
    };

    for (const auto& g : raws) {
        float p[3] = {g.x, g.y, g.z};
        float pc[3];
        pc[0] = cam.w[0]*p[0] + cam.w[1]*p[1] + cam.w[2]*p[2] + cam.t[0];
        pc[1] = cam.w[3]*p[0] + cam.w[4]*p[1] + cam.w[5]*p[2] + cam.t[1];
        pc[2] = cam.w[6]*p[0] + cam.w[7]*p[1] + cam.w[8]*p[2] + cam.t[2];
        if (pc[2] < 0.1f) { n_culled_behind++; continue; }
        if (pc[2] > 100.0f) { n_culled_far++; continue; }

        float mx = cam.fx * pc[0] / pc[2] + cam.cx;
        float my = cam.fy * pc[1] / pc[2] + cam.cy;

        float sigma_w[9]; cov3d(g.rot, g.scale, sigma_w);
        float Wt[9]; mat3_transpose(cam.w, Wt);
        float tmp[9]; mat3_mul(cam.w, sigma_w, tmp);
        float sigma_cam[9]; mat3_mul(tmp, Wt, sigma_cam);

        float z = pc[2];
        float z2 = z*z;
        float j00 = cam.fx / z;
        float j02 = -cam.fx * pc[0] / z2;
        float j11 = cam.fy / z;
        float j12 = -cam.fy * pc[1] / z2;
        // M = J * sigma_cam (2x3): row0 = j00*r0 + j02*r2; row1 = j11*r1 + j12*r2
        float r0[3] = {sigma_cam[0], sigma_cam[1], sigma_cam[2]};
        float r1[3] = {sigma_cam[3], sigma_cam[4], sigma_cam[5]};
        float r2[3] = {sigma_cam[6], sigma_cam[7], sigma_cam[8]};
        float m0[3] = {j00*r0[0] + j02*r2[0], j00*r0[1] + j02*r2[1], j00*r0[2] + j02*r2[2]};
        float m1[3] = {j11*r1[0] + j12*r2[0], j11*r1[1] + j12*r2[1], j11*r1[2] + j12*r2[2]};
        // Sigma_2d = M * J^T (2x2). J^T rows: [j00,0], [0,j11], [j02,j12].
        float a = m0[0]*j00 + m0[1]*0.0f  + m0[2]*j02;
        float b = m0[0]*0.0f + m0[1]*j11  + m0[2]*j12;
        float c = m1[0]*0.0f + m1[1]*j11  + m1[2]*j12;

        float a_aa = a + 0.3f;
        float c_aa = c + 0.3f;
        float b_aa = b;

        float det = a_aa*c_aa - b_aa*b_aa;
        if (!(det > 0.0f) || !isfinite(det)) { n_culled_bad_cov++; continue; }
        float inv_det = 1.0f / det;
        float cxx =  c_aa * inv_det;
        float cxy = -b_aa * inv_det;
        float cyy =  a_aa * inv_det;

        // SH evaluation (degree 0 or 3). View direction in WORLD space, from
        // camera origin to gaussian center.
        float vdx = g.x - cam_origin[0];
        float vdy = g.y - cam_origin[1];
        float vdz = g.z - cam_origin[2];
        float vdn = sqrtf(vdx*vdx + vdy*vdy + vdz*vdz);
        if (vdn < 1e-8f) vdn = 1e-8f;
        float vx = vdx / vdn, vy = vdy / vdn, vz = vdz / vdn;
        float rr, gg, bb;
        sh_to_rgb(g.f_dc, g.f_rest, vx, vy, vz, &rr, &gg, &bb);
        auto clamp01 = [](float v){ return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };
        float col_r = clamp01(rr);
        float col_g = clamp01(gg);
        float col_b = clamp01(bb);
        float op = sigmoidf(g.opacity_logit);

        if (pc[2] < depth_min) depth_min = pc[2];
        if (pc[2] > depth_max) depth_max = pc[2];
        if (mx < m2d_min_x) m2d_min_x = mx;
        if (my < m2d_min_y) m2d_min_y = my;
        if (mx > m2d_max_x) m2d_max_x = mx;
        if (my > m2d_max_y) m2d_max_y = my;
        conic_scales.push_back((a_aa + c_aa) * 0.5f);

        ProjGaussian pg;
        pg.mx = mx; pg.my = my;
        pg.cxx = cxx; pg.cxy = cxy; pg.cyy = cyy;
        pg.opacity = op;
        pg.r = col_r; pg.g = col_g; pg.b = col_b;
        pg.depth = pc[2];
        out.push_back(pg);
    }

    std::sort(out.begin(), out.end(),
              [](const ProjGaussian& a, const ProjGaussian& b){ return a.depth < b.depth; });

    // Dump pre-sort per-gaussian colors for cross-check vs CPU SH reference.
    // (We're inside project_all; we already did std::sort above so the order
    // is by depth, not PLY-order. To compare to a PLY-ordered CPU reference
    // we need to dump from a parallel array. We can't, easily, after the
    // sort. Instead, we emit a separate PLY-ordered color file in main()
    // by calling sh_to_rgb once per gaussian without going through project.)

    std::sort(conic_scales.begin(), conic_scales.end());
    float cs_med = conic_scales.empty() ? 0.0f : conic_scales[conic_scales.size()/2];

    if (stats_out) {
        stats_out->n_total = raws.size();
        stats_out->n_projected = out.size();
        stats_out->n_culled_behind = n_culled_behind;
        stats_out->n_culled_far = n_culled_far;
        stats_out->n_culled_bad_cov = n_culled_bad_cov;
        stats_out->depth_min = depth_min;
        stats_out->depth_max = depth_max;
        stats_out->mean2d_min_x = m2d_min_x; stats_out->mean2d_min_y = m2d_min_y;
        stats_out->mean2d_max_x = m2d_max_x; stats_out->mean2d_max_y = m2d_max_y;
        stats_out->conic_scale_median = cs_med;
    }
    return out;
}

// ---------- Output ----------

static void save_ppm(const std::string& path,
                     const std::vector<float>& pr,
                     const std::vector<float>& pg,
                     const std::vector<float>& pb,
                     int w, int h) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "create ppm %s\n", path.c_str()); exit(1); }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    size_t n = (size_t)w * (size_t)h;
    std::vector<unsigned char> buf(n*3);
    for (size_t i = 0; i < n; ++i) {
        auto to_u8 = [](float v){
            v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
            int iv = (int)(v * 255.0f + 0.5f);
            if (iv < 0) iv = 0; if (iv > 255) iv = 255;
            return (unsigned char)iv;
        };
        buf[i*3+0] = to_u8(pr[i]);
        buf[i*3+1] = to_u8(pg[i]);
        buf[i*3+2] = to_u8(pb[i]);
    }
    fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);
}

// ---------- Render driver ----------

struct Timings {
    double project_ms;
    double sort_included_in_project_ms; // sort is inside project_all
    double h2d_ms;
    double kernel_iters_ms[ITERS];
    double kernel_median_ms;
    double d2h_ms;
    int n_projected;
};

static Timings render_cam(const std::vector<RawGaussian>& raws,
                          const CamPose& cam,
                          const char* label,
                          const std::string& out_ppm,
                          FILE* csv) {
    Timings t{};
    printf("=== [%s] ===\n", label);
    auto tp0 = std::chrono::high_resolution_clock::now();
    ProjStats stats;
    auto proj = project_all(raws, cam, &stats);
    auto tp1 = std::chrono::high_resolution_clock::now();
    t.project_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("  project: total=%zu projected=%zu culled_behind=%zu culled_far=%zu culled_bad_cov=%zu\n",
           stats.n_total, stats.n_projected, stats.n_culled_behind, stats.n_culled_far, stats.n_culled_bad_cov);
    printf("  depth_range=[%.3f, %.3f]\n", stats.depth_min, stats.depth_max);
    printf("  mean2d_range=[(%.1f,%.1f)..(%.1f,%.1f)]\n",
           stats.mean2d_min_x, stats.mean2d_min_y, stats.mean2d_max_x, stats.mean2d_max_y);
    printf("  conic_scale_median=%.4f\n", stats.conic_scale_median);
    printf("  projection+sort (host): %.3f ms\n", t.project_ms);

    if (proj.empty()) {
        printf("  [%s] no gaussians; skipping\n", label);
        t.n_projected = 0;
        return t;
    }
    int n = (int)proj.size();
    t.n_projected = n;

    std::vector<float> mx(n), my(n), cxx(n), cxy(n), cyy(n), op(n), cr(n), cg(n), cb(n);
    for (int i = 0; i < n; ++i) {
        mx[i] = proj[i].mx; my[i] = proj[i].my;
        cxx[i] = proj[i].cxx; cxy[i] = proj[i].cxy; cyy[i] = proj[i].cyy;
        op[i] = proj[i].opacity;
        cr[i] = proj[i].r; cg[i] = proj[i].g; cb[i] = proj[i].b;
    }

    // Device buffers
    float *d_mx, *d_my, *d_cxx, *d_cxy, *d_cyy, *d_op, *d_cr, *d_cg, *d_cb;
    float *d_or, *d_og, *d_ob;
    size_t gn = (size_t)n * sizeof(float);
    size_t pn = (size_t)W * (size_t)H * sizeof(float);
    CK(cudaMalloc(&d_mx, gn));  CK(cudaMalloc(&d_my, gn));
    CK(cudaMalloc(&d_cxx, gn)); CK(cudaMalloc(&d_cxy, gn)); CK(cudaMalloc(&d_cyy, gn));
    CK(cudaMalloc(&d_op, gn));
    CK(cudaMalloc(&d_cr, gn));  CK(cudaMalloc(&d_cg, gn));  CK(cudaMalloc(&d_cb, gn));
    CK(cudaMalloc(&d_or, pn));  CK(cudaMalloc(&d_og, pn));  CK(cudaMalloc(&d_ob, pn));

    cudaEvent_t evs, eve;
    CK(cudaEventCreate(&evs));
    CK(cudaEventCreate(&eve));

    // H2D timing
    CK(cudaEventRecord(evs));
    CK(cudaMemcpy(d_mx, mx.data(), gn, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_my, my.data(), gn, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cxx, cxx.data(), gn, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cxy, cxy.data(), gn, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cyy, cyy.data(), gn, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_op, op.data(), gn, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cr, cr.data(), gn, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cg, cg.data(), gn, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cb, cb.data(), gn, cudaMemcpyHostToDevice));
    CK(cudaMemsetAsync(d_or, 0, pn));
    CK(cudaMemsetAsync(d_og, 0, pn));
    CK(cudaMemsetAsync(d_ob, 0, pn));
    CK(cudaEventRecord(eve));
    CK(cudaEventSynchronize(eve));
    float h2d_ms = 0.0f;
    CK(cudaEventElapsedTime(&h2d_ms, evs, eve));
    t.h2d_ms = h2d_ms;
    printf("  H2D copy: %.3f ms\n", h2d_ms);

    dim3 block(BS, BS);
    dim3 grid((W + BS - 1) / BS, (H + BS - 1) / BS);

    // Warmup
    rasterize_2dgs<<<grid, block>>>(d_mx, d_my, d_cxx, d_cxy, d_cyy, d_op,
                                     d_cr, d_cg, d_cb, n, W, H, d_or, d_og, d_ob);
    CK(cudaDeviceSynchronize());

    // Timed iters
    double times[ITERS];
    for (int it = 0; it < ITERS; ++it) {
        CK(cudaEventRecord(evs));
        rasterize_2dgs<<<grid, block>>>(d_mx, d_my, d_cxx, d_cxy, d_cyy, d_op,
                                         d_cr, d_cg, d_cb, n, W, H, d_or, d_og, d_ob);
        CK(cudaEventRecord(eve));
        CK(cudaEventSynchronize(eve));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, evs, eve));
        times[it] = ms;
        t.kernel_iters_ms[it] = ms;
        printf("  kernel iter %d: %.3f ms\n", it, ms);
        if (csv) fprintf(csv, "cuda-3dgs-real,rasterize_2dgs,%s,%d,%d,%.6f\n",
                         label, n, it, ms);
    }
    double sorted[ITERS];
    for (int i=0;i<ITERS;++i) sorted[i]=times[i];
    std::sort(sorted, sorted+ITERS);
    t.kernel_median_ms = sorted[ITERS/2];
    printf("  median kernel time: %.3f ms\n", t.kernel_median_ms);

    // D2H
    auto td0 = std::chrono::high_resolution_clock::now();
    std::vector<float> hr(W*H), hg(W*H), hb(W*H);
    CK(cudaMemcpy(hr.data(), d_or, pn, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hg.data(), d_og, pn, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(), d_ob, pn, cudaMemcpyDeviceToHost));
    auto td1 = std::chrono::high_resolution_clock::now();
    t.d2h_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();
    printf("  D2H copy: %.3f ms\n", t.d2h_ms);

    // sanity: nonzero pixel count
    size_t nonzero = 0;
    for (size_t i = 0; i < hr.size(); ++i)
        if (hr[i] > 0.01f || hg[i] > 0.01f || hb[i] > 0.01f) ++nonzero;
    printf("  nonzero pixels: %zu/%zu = %.2f%%\n",
           nonzero, hr.size(), 100.0 * (double)nonzero / (double)hr.size());

    save_ppm(out_ppm, hr, hg, hb, W, H);
    printf("  wrote %s\n", out_ppm.c_str());

    // Dump raw f32 RGB planes for byte-exact cross-impl comparison (cam A only).
    if (strcmp(label, "camA_minusZ") == 0) {
        FILE* fr = fopen("output_utsuho_plush_A.f32", "wb");
        if (fr) {
            fwrite(hr.data(), 4, hr.size(), fr);
            fwrite(hg.data(), 4, hg.size(), fr);
            fwrite(hb.data(), 4, hb.size(), fr);
            fclose(fr);
            printf("  wrote output_utsuho_plush_A.f32 (%zu floats)\n", hr.size()*3);
        }
    }

    cudaFree(d_mx); cudaFree(d_my);
    cudaFree(d_cxx); cudaFree(d_cxy); cudaFree(d_cyy);
    cudaFree(d_op);
    cudaFree(d_cr); cudaFree(d_cg); cudaFree(d_cb);
    cudaFree(d_or); cudaFree(d_og); cudaFree(d_ob);
    cudaEventDestroy(evs); cudaEventDestroy(eve);
    return t;
}

// ---------- Main ----------

int main(int argc, char** argv) {
    std::string ply_path = (argc > 1) ? argv[1]
        : "../oxide-3dgs-real/scenes/utsuho_plush.ply";
    printf("Loading %s\n", ply_path.c_str());
    auto raws = parse_ply(ply_path);
    printf("Parsed %zu gaussians\n", raws.size());

    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[cuda] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    // Scene stats
    float cx = 0, cy = 0, cz = 0;
    float mn[3] = {INFINITY, INFINITY, INFINITY};
    float mx[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (const auto& g : raws) {
        cx += g.x; cy += g.y; cz += g.z;
        if (g.x < mn[0]) mn[0]=g.x; if (g.x > mx[0]) mx[0]=g.x;
        if (g.y < mn[1]) mn[1]=g.y; if (g.y > mx[1]) mx[1]=g.y;
        if (g.z < mn[2]) mn[2]=g.z; if (g.z > mx[2]) mx[2]=g.z;
    }
    float nf = (float)raws.size();
    cx /= nf; cy /= nf; cz /= nf;
    printf("Scene centroid: (%.3f, %.3f, %.3f)\n", cx, cy, cz);
    printf("Scene bbox   : min=(%.3f,%.3f,%.3f) max=(%.3f,%.3f,%.3f)\n",
           mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]);
    float extent = sqrtf((mx[0]-mn[0])*(mx[0]-mn[0])
                       + (mx[1]-mn[1])*(mx[1]-mn[1])
                       + (mx[2]-mn[2])*(mx[2]-mn[2]));
    printf("Scene diag   : %.3f\n", extent);

    float fx = 800.0f, fy = 800.0f, cx_p = 400.0f, cy_p = 400.0f;
    float dist = extent * 1.5f;

    CamPose camA;
    {
        float I[9] = {1,0,0, 0,1,0, 0,0,1};
        memcpy(camA.w, I, sizeof(I));
        camA.t[0] = -cx; camA.t[1] = -cy; camA.t[2] = -(cz - dist);
        camA.fx = fx; camA.fy = fy; camA.cx = cx_p; camA.cy = cy_p;
    }
    CamPose camB;
    {
        float I[9] = {1,0,0, 0,1,0, 0,0,1};
        memcpy(camB.w, I, sizeof(I));
        camB.t[0] = -cx; camB.t[1] = -cy; camB.t[2] = -(cz + dist);
        camB.fx = fx; camB.fy = fy; camB.cx = cx_p; camB.cy = cy_p;
    }
    CamPose camC;
    {
        float Fy[9] = {1,0,0, 0,-1,0, 0,0,1};
        memcpy(camC.w, Fy, sizeof(Fy));
        camC.t[0] = -cx; camC.t[1] = cy; camC.t[2] = -(cz - dist);
        camC.fx = fx; camC.fy = fy; camC.cx = cx_p; camC.cy = cy_p;
    }
    CamPose camD;
    {
        float Ry180[9] = {-1,0,0, 0,1,0, 0,0,-1};
        memcpy(camD.w, Ry180, sizeof(Ry180));
        camD.t[0] = cx; camD.t[1] = -cy; camD.t[2] = cz + dist;
        camD.fx = fx; camD.fy = fy; camD.cx = cx_p; camD.cy = cy_p;
    }

    FILE* csv = fopen("results.csv", "w");
    if (csv) fprintf(csv, "impl,kernel,camera,n_projected,iter,gpu_ms\n");

    // Render all 4 cameras (like Rust does).
    Timings tA = render_cam(raws, camA, "camA_minusZ",   "output_utsuho_plush_A.ppm", csv);
    Timings tB = render_cam(raws, camB, "camB_plusZ_noflip", "output_utsuho_plush_B.ppm", csv);
    Timings tC = render_cam(raws, camC, "camC_flipY",    "output_utsuho_plush_C.ppm", csv);
    Timings tD = render_cam(raws, camD, "camD_roty180",  "output_utsuho_plush_D.ppm", csv);

    // Canonical (unsuffixed) PPM = camera A, matching oxide-3dgs-real convention.
    // Easiest: re-render A, but actually just symlink; since camA was already
    // written, the reader expects output_utsuho_plush.ppm too — copy bytes.
    {
        FILE* src = fopen("output_utsuho_plush_A.ppm", "rb");
        FILE* dst = fopen("output_utsuho_plush.ppm", "wb");
        if (src && dst) {
            unsigned char buf[65536];
            size_t r;
            while ((r = fread(buf, 1, sizeof(buf), src)) > 0) fwrite(buf, 1, r, dst);
        }
        if (src) fclose(src);
        if (dst) fclose(dst);
        printf("wrote canonical output_utsuho_plush.ppm (= cam A)\n");
    }

    printf("\n===== SUMMARY =====\n");
    printf("cam   n_proj   proj_ms   h2d_ms   median_kernel_ms\n");
    auto row = [](const char* lbl, const Timings& t){
        printf("%-5s %7d  %8.3f  %7.3f  %10.3f\n",
               lbl, t.n_projected, t.project_ms, t.h2d_ms, t.kernel_median_ms);
    };
    row("A", tA); row("B", tB); row("C", tC); row("D", tD);

    if (csv) fclose(csv);
    return 0;
}
