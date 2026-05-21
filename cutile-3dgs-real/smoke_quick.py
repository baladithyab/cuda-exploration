"""Quick standalone smoke test: 5 gaussians, single 16x16 tile, manual data."""
from __future__ import annotations

import numpy as np
import cuda.tile as ct
import cupy

import sys
sys.path.insert(0, ".")
from rasterize import make_rasterize_kernel

W = 16
H = 16
BS = 16

print("Building kernel...")
kernel = make_rasterize_kernel(W, H, BS)
print("OK kernel built")

# 3 gaussians: one at (4,4), one at (8,8), one at (12,12).
n = 3
mx = np.array([4.0, 8.0, 12.0], dtype=np.float32)
my = np.array([4.0, 8.0, 12.0], dtype=np.float32)
cxx = np.array([0.5, 0.5, 0.5], dtype=np.float32)
cxy = np.array([0.0, 0.0, 0.0], dtype=np.float32)
cyy = np.array([0.5, 0.5, 0.5], dtype=np.float32)
opacity = np.array([0.9, 0.9, 0.9], dtype=np.float32)
cr = np.array([1.0, 0.0, 0.0], dtype=np.float32)
cg = np.array([0.0, 1.0, 0.0], dtype=np.float32)
cb = np.array([0.0, 0.0, 1.0], dtype=np.float32)

d_mx = cupy.asarray(mx)
d_my = cupy.asarray(my)
d_cxx = cupy.asarray(cxx)
d_cxy = cupy.asarray(cxy)
d_cyy = cupy.asarray(cyy)
d_op = cupy.asarray(opacity)
d_cr = cupy.asarray(cr)
d_cg = cupy.asarray(cg)
d_cb = cupy.asarray(cb)
d_or = cupy.zeros((H, W), dtype=cupy.float32)
d_og = cupy.zeros((H, W), dtype=cupy.float32)
d_ob = cupy.zeros((H, W), dtype=cupy.float32)

stream = cupy.cuda.get_current_stream()
print("Launching kernel...")
ct.launch(stream.ptr, (W // BS, H // BS), kernel,
          (d_mx, d_my, d_cxx, d_cxy, d_cyy, d_op, d_cr, d_cg, d_cb,
           int(n), d_or, d_og, d_ob))
stream.synchronize()
print("OK kernel ran")

hr = d_or.get(); hg = d_og.get(); hb = d_ob.get()
print(f"red @ (4,4): {hr[4,4]:.3f} (expect ~0.9)")
print(f"red @ (8,8): {hr[8,8]:.3f} (expect ~0)")
print(f"green @ (8,8): {hg[8,8]:.3f}")
print(f"blue @ (12,12): {hb[12,12]:.3f}")
print(f"red col sum: {hr.sum():.3f}")
print(f"green col sum: {hg.sum():.3f}")
print(f"blue col sum: {hb.sum():.3f}")
print("Done.")
