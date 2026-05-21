# Wave 22.1 API probe — what symbols does std.gpu.sync expose?

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

# Try each candidate import individually and see which compiles:
from std.gpu.sync import cp_async_bulk_tensor


def probe_kernel():
    var tid = thread_idx.x
    if tid == 0:
        print("[w22.1-probe] kernel ran")


def main() raises -> None:
    if not has_accelerator():
        return

    with DeviceContext() as ctx:
        ctx.enqueue_function[probe_kernel, probe_kernel](
            grid_dim=1, block_dim=1,
        )
        ctx.synchronize()
        print("[w22.1-probe] cp_async_bulk_tensor IS available!")
