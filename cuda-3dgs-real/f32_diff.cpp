// Compare two raw f32 buffers byte-by-byte and report f32-level error stats.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>

static bool read_f32(const std::string& path, std::vector<float>* out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "open %s\n", path.c_str()); return false; }
    auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if ((size_t)sz % 4 != 0) { fprintf(stderr, "size not multiple of 4: %s\n", path.c_str()); return false; }
    out->resize((size_t)sz / 4);
    f.read((char*)out->data(), sz);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s a.f32 b.f32\n", argv[0]); return 1; }
    std::vector<float> a, b;
    if (!read_f32(argv[1], &a)) return 1;
    if (!read_f32(argv[2], &b)) return 1;
    if (a.size() != b.size()) {
        fprintf(stderr, "size mismatch: %zu vs %zu\n", a.size(), b.size());
        return 1;
    }
    size_t n = a.size();
    size_t bit_identical = 0;
    size_t value_eq = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double sse = 0.0;
    size_t at_max_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        // bit identity
        uint32_t ai, bi;
        memcpy(&ai, &a[i], 4); memcpy(&bi, &b[i], 4);
        if (ai == bi) ++bit_identical;
        if (a[i] == b[i]) ++value_eq;
        double d = (double)a[i] - (double)b[i];
        double ad = fabs(d);
        if (ad > max_abs) { max_abs = ad; at_max_idx = i; }
        double denom = std::max(fabs((double)a[i]), fabs((double)b[i]));
        if (denom > 1e-12) {
            double rel = ad / denom;
            if (rel > max_rel) max_rel = rel;
        }
        sse += d*d;
    }
    double rmse = sqrt(sse / (double)n);
    printf("F32 compare %s vs %s\n", argv[1], argv[2]);
    printf("  values:           %zu\n", n);
    printf("  bit-identical:    %zu (%.4f%%)\n", bit_identical, 100.0 * bit_identical / n);
    printf("  value-equal:      %zu (%.4f%%)\n", value_eq, 100.0 * value_eq / n);
    printf("  max_abs_err:      %.6e\n", max_abs);
    printf("  max_rel_err:      %.6e\n", max_rel);
    printf("  rmse:             %.6e\n", rmse);
    if (max_abs > 0) {
        printf("  at index %zu: a=%.9g b=%.9g\n", at_max_idx, a[at_max_idx], b[at_max_idx]);
    }
    // Spec: max_abs_err <= 1e-3 per-pixel f32
    if (max_abs <= 1e-3) {
        printf("  RESULT: PASS (max_abs_err %.3e ≤ 1e-3)\n", max_abs);
        return 0;
    } else {
        printf("  RESULT: FAIL (max_abs_err %.3e > 1e-3)\n", max_abs);
        return 2;
    }
}
