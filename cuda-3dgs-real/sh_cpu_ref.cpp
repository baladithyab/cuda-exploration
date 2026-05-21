// Standalone independent CPU SH degree 3 reference. Evaluates SH on the
// utsuho_plush.ply for one camera pose (camA) and dumps per-gaussian RGB
// to stdout as binary float triples. Used to cross-check the inline SH
// implementations in oxide-3dgs-real and cuda-3dgs-real.
//
// Code style is intentionally different from rasterize.cu (separate-vector
// SH coefficient layout, different summation order) to act as an
// independent reference.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

struct G {
    float x, y, z;
    float dc[3];
    float rest[45];
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s scene.ply out.f32\n", argv[0]);
        return 1;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { fprintf(stderr, "open\n"); return 1; }
    std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    const char* needle = "end_header\n";
    size_t nlen = strlen(needle);
    size_t hend = 0;
    for (size_t i = 0; i + nlen <= buf.size(); ++i)
        if (memcmp(&buf[i], needle, nlen) == 0) { hend = i + nlen; break; }
    std::string header(&buf[0], hend);
    size_t nverts = 0;
    std::vector<std::string> props;
    {
        size_t p = 0;
        while (p < header.size()) {
            size_t e = header.find('\n', p);
            if (e == std::string::npos) e = header.size();
            std::string line = header.substr(p, e - p);
            p = e + 1;
            if (line.rfind("element vertex ", 0) == 0)
                nverts = (size_t)atoll(line.c_str() + 15);
            else if (line.rfind("property float ", 0) == 0) {
                std::string name = line.substr(15);
                while (!name.empty() && (name.back()=='\r' || name.back()==' '))
                    name.pop_back();
                props.push_back(name);
            }
        }
    }
    auto idx = [&](const char* n)->int {
        for (size_t k = 0; k < props.size(); ++k) if (props[k] == n) return (int)k;
        return -1;
    };
    int ix = idx("x"), iy = idx("y"), iz = idx("z");
    int idc0 = idx("f_dc_0"), idc1 = idx("f_dc_1"), idc2 = idx("f_dc_2");
    int irest[45];
    for (int k = 0; k < 45; ++k) {
        char nm[16]; snprintf(nm, sizeof(nm), "f_rest_%d", k);
        irest[k] = idx(nm);
        if (irest[k] < 0) { fprintf(stderr, "no SH3\n"); return 1; }
    }
    const char* body = &buf[hend];
    size_t nprops = props.size();
    std::vector<G> gs(nverts);
    for (size_t i = 0; i < nverts; ++i) {
        size_t base = i * nprops * 4;
        auto rf = [&](int p) { float v; memcpy(&v, body + base + p*4, 4); return v; };
        gs[i].x = rf(ix); gs[i].y = rf(iy); gs[i].z = rf(iz);
        gs[i].dc[0] = rf(idc0); gs[i].dc[1] = rf(idc1); gs[i].dc[2] = rf(idc2);
        for (int k = 0; k < 45; ++k) gs[i].rest[k] = rf(irest[k]);
    }

    // Replicate camA setup (must match oxide-3dgs-real/cuda-3dgs-real exactly):
    //   centroid + extent, dist = extent*1.5, W = I, t = -origin where
    //   origin = (cx, cy, cz - dist).
    float cx=0, cy=0, cz=0;
    float mn[3]={INFINITY,INFINITY,INFINITY}, mx[3]={-INFINITY,-INFINITY,-INFINITY};
    for (auto& g : gs) {
        cx += g.x; cy += g.y; cz += g.z;
        if (g.x<mn[0]) mn[0]=g.x; if (g.x>mx[0]) mx[0]=g.x;
        if (g.y<mn[1]) mn[1]=g.y; if (g.y>mx[1]) mx[1]=g.y;
        if (g.z<mn[2]) mn[2]=g.z; if (g.z>mx[2]) mx[2]=g.z;
    }
    cx /= nverts; cy /= nverts; cz /= nverts;
    float ext = sqrtf((mx[0]-mn[0])*(mx[0]-mn[0])
                    + (mx[1]-mn[1])*(mx[1]-mn[1])
                    + (mx[2]-mn[2])*(mx[2]-mn[2]));
    float dist = ext * 1.5f;
    // camA: W = I, t = (-cx, -cy, -(cz - dist)). origin = -W^T t = (cx, cy, cz - dist).
    float origin[3] = {cx, cy, cz - dist};
    fprintf(stderr, "scene: nverts=%zu, centroid=(%.3f,%.3f,%.3f), dist=%.3f\n",
            nverts, cx, cy, cz, dist);
    fprintf(stderr, "cam origin: (%.3f,%.3f,%.3f)\n", origin[0], origin[1], origin[2]);

    // Independent SH3 evaluator. We use *struct-of-coefs by band* layout
    // (bandk_chc) instead of the flat 15-rest-per-channel layout used in
    // the renderers. This is intentionally different to catch indexing bugs.
    //
    //   Inria PLY:
    //     ch0 (R): rest[ 0..14] = [b1_0..b1_2, b2_0..b2_4, b3_0..b3_6]
    //     ch1 (G): rest[15..29] = same layout
    //     ch2 (B): rest[30..44] = same layout
    //
    //   Re-pack into per-(band, m, channel):
    //     Y1_-1 ↔ rest[ 0+ch*15]   coeff for direction -y
    //     Y1_0  ↔ rest[ 1+ch*15]   coeff for direction  z
    //     Y1_1  ↔ rest[ 2+ch*15]   coeff for direction -x
    //     Y2_-2 ↔ rest[ 3+ch*15]
    //     Y2_-1 ↔ rest[ 4+ch*15]
    //     Y2_0  ↔ rest[ 5+ch*15]
    //     Y2_1  ↔ rest[ 6+ch*15]
    //     Y2_2  ↔ rest[ 7+ch*15]
    //     Y3_-3..Y3_3 ↔ rest[ 8+ch*15] .. rest[14+ch*15]
    //
    // Real spherical harmonics (Inria/gsplat convention):
    //   Y0_0  =  C0
    //   Y1_-1 = -C1*y;  Y1_0 =  C1*z;   Y1_1 = -C1*x
    //   Y2_-2 =  C2_0 * x*y                     (sqrt(15/pi)/2)
    //   Y2_-1 =  C2_1 * y*z
    //   Y2_0  =  C2_2 * (2*z^2 - x^2 - y^2)
    //   Y2_1  =  C2_3 * x*z
    //   Y2_2  =  C2_4 * (x^2 - y^2)
    //   Y3_-3 =  C3_0 * y*(3*x^2 - y^2)
    //   Y3_-2 =  C3_1 * x*y*z
    //   Y3_-1 =  C3_2 * y*(4*z^2 - x^2 - y^2)
    //   Y3_0  =  C3_3 * z*(2*z^2 - 3*x^2 - 3*y^2)
    //   Y3_1  =  C3_4 * x*(4*z^2 - x^2 - y^2)
    //   Y3_2  =  C3_5 * z*(x^2 - y^2)
    //   Y3_3  =  C3_6 * x*(x^2 - 3*y^2)
    //
    const double C0  =  0.28209479177387814;
    const double C1  =  0.4886025119029199;
    const double C2_0=  1.0925484305920792;
    const double C2_1= -1.0925484305920792;
    const double C2_2=  0.31539156525252005;
    const double C2_3= -1.0925484305920792;
    const double C2_4=  0.5462742152960396;
    const double C3_0= -0.5900435899266435;
    const double C3_1=  2.890611442640554;
    const double C3_2= -0.4570457994644658;
    const double C3_3=  0.3731763325901154;
    const double C3_4= -0.4570457994644658;
    const double C3_5=  1.445305721320277;
    const double C3_6= -0.5900435899266435;

    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { fprintf(stderr, "create out\n"); return 1; }

    // Per-gaussian SH eval. Output is (R, G, B) f32 per gaussian, in PLY order.
    for (auto& g : gs) {
        double dx = g.x - origin[0], dy = g.y - origin[1], dz = g.z - origin[2];
        double dn = sqrt(dx*dx + dy*dy + dz*dz);
        if (dn < 1e-8) dn = 1e-8;
        double x = dx/dn, y = dy/dn, z = dz/dn;

        // Compute basis values once (in double for reference quality):
        double Y0 = C0;
        double Y1m = -C1 * y;
        double Y10 =  C1 * z;
        double Y1p = -C1 * x;
        double xx = x*x, yy = y*y, zz = z*z;
        double Y2m2 = C2_0 * x*y;
        double Y2m1 = C2_1 * y*z;
        double Y20  = C2_2 * (2.0*zz - xx - yy);
        double Y21  = C2_3 * x*z;
        double Y22  = C2_4 * (xx - yy);
        double Y3m3 = C3_0 * y*(3.0*xx - yy);
        double Y3m2 = C3_1 * x*y*z;
        double Y3m1 = C3_2 * y*(4.0*zz - xx - yy);
        double Y30  = C3_3 * z*(2.0*zz - 3.0*xx - 3.0*yy);
        double Y31  = C3_4 * x*(4.0*zz - xx - yy);
        double Y32  = C3_5 * z*(xx - yy);
        double Y33  = C3_6 * x*(xx - 3.0*yy);

        float rgb[3];
        for (int ch = 0; ch < 3; ++ch) {
            const float* r = g.rest + ch*15;
            double v = Y0  * (double)g.dc[ch]
                     + Y1m * (double)r[0]
                     + Y10 * (double)r[1]
                     + Y1p * (double)r[2]
                     + Y2m2* (double)r[3]
                     + Y2m1* (double)r[4]
                     + Y20 * (double)r[5]
                     + Y21 * (double)r[6]
                     + Y22 * (double)r[7]
                     + Y3m3* (double)r[8]
                     + Y3m2* (double)r[9]
                     + Y3m1* (double)r[10]
                     + Y30 * (double)r[11]
                     + Y31 * (double)r[12]
                     + Y32 * (double)r[13]
                     + Y33 * (double)r[14];
            v += 0.5;
            // Match the renderers: clamp to [0,1].
            if (v < 0.0) v = 0.0;
            if (v > 1.0) v = 1.0;
            rgb[ch] = (float)v;
        }
        fwrite(rgb, 4, 3, fout);
    }
    fclose(fout);
    fprintf(stderr, "wrote %s (%zu gaussians × 3 = %zu floats)\n",
            argv[2], nverts, nverts*3);
    return 0;
}
