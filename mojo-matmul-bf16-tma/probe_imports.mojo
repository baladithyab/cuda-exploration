# Wave 22.1 API probe — does Mojo 1.0.0b1 have a cp_async_bulk API?
# This is a 5-minute probe to check what TMA-relevant primitives exist
# in std.gpu.* before committing to a full mojo-matmul-bf16-tma cell.

from sys import has_accelerator
from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

# Try import paths that MIGHT have cp_async_bulk:
from std.gpu.sync import *  # if cp_async_bulk lives under sync


def probe_kernel():
    var tid = thread_idx.x
    if tid == 0:
        print("[w22.1-probe] kernel ran on tid=0")


def main() raises -> None:
    if not has_accelerator():
        print("[w22.1-probe] No GPU accelerator; aborting.")
        return

    with DeviceContext() as ctx:
        ctx.enqueue_function[probe_kernel, probe_kernel](
            grid_dim=1, block_dim=1,
        )
        ctx.synchronize()
        print("[w22.1-probe] OK")
