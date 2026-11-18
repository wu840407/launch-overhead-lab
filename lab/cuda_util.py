"""Compile CUDA C with NVRTC and launch it through the CUDA driver API on PyTorch's stream.

No nvcc is needed: NVRTC and the driver bindings come from the `cuda-python` package, and
PyTorch provides device memory, reference results and CUDA graph capture.
"""
from __future__ import annotations

import ctypes
import site
from pathlib import Path

import torch
from cuda.bindings import driver, nvrtc

KERNELS = Path(__file__).with_name("kernels.cu")


def _check(ret):
    err, *rest = ret
    if int(err) != 0:
        raise RuntimeError(f"CUDA/NVRTC error: {err}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else tuple(rest)


def _include_dirs() -> list[str]:
    """Directories in the installed NVIDIA wheels that contain cuda_fp16.h."""
    found = []
    for base in site.getsitepackages():
        for header in Path(base, "nvidia").glob("*/include/cuda_fp16.h"):
            found.append(str(header.parent))
    if not found:
        raise RuntimeError("cuda_fp16.h not found in the installed nvidia-* packages")
    return found


class Module:
    """A compiled .cu file. Kernels are looked up by their extern "C" name."""

    def __init__(self, path: Path = KERNELS):
        torch.zeros(1, device="cuda")  # make PyTorch's primary context the current context
        src = Path(path).read_text(encoding="utf-8")
        prog = _check(nvrtc.nvrtcCreateProgram(src.encode(), Path(path).name.encode(), 0, [], []))
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17"]
        opts += [f"-I{d}".encode() for d in _include_dirs()]
        ret = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if int(ret[0]) != 0:
            size = _check(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(log.decode(errors="replace"))
        size = _check(nvrtc.nvrtcGetCUBINSize(prog))
        cubin = b" " * size
        _check(nvrtc.nvrtcGetCUBIN(prog, cubin) + (None,))
        self._module = _check(driver.cuModuleLoadData(cubin))
        self._functions: dict[str, object] = {}

    def function(self, name: str):
        if name not in self._functions:
            self._functions[name] = _check(driver.cuModuleGetFunction(self._module, name.encode()))
        return self._functions[name]

    def launcher(self, name: str, grid, block, args, shared_bytes: int = 0) -> "Launch":
        return Launch(self.function(name), grid, block, args, shared_bytes)


class Launch:
    """A kernel launch with its arguments packed once.

    Calling it issues exactly one cuLaunchKernel, so a loop of calls measures the cost of
    launching from Python through the driver API and nothing else.
    Arguments: torch tensors (passed as device pointers) or ctypes scalars.
    """

    def __init__(self, fn, grid, block, args, shared_bytes: int = 0):
        self.fn = fn
        self.grid = tuple(grid) + (1,) * (3 - len(grid))
        self.block = tuple(block) + (1,) * (3 - len(block))
        self.shared_bytes = shared_bytes
        self._keep = args  # keep tensors alive for as long as the launch exists
        self._args = [ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else a for a in args]
        for a in self._args:
            if not isinstance(a, ctypes._SimpleCData):
                raise TypeError(f"argument must be a tensor or a ctypes scalar, got {type(a).__name__}")
        self._params = (ctypes.c_void_p * max(len(self._args), 1))(
            *[ctypes.addressof(a) for a in self._args]
        )
        self._params_ptr = ctypes.addressof(self._params) if self._args else 0

    def __call__(self, stream: int) -> None:
        err, = driver.cuLaunchKernel(self.fn, *self.grid, *self.block, self.shared_bytes,
                                     stream, self._params_ptr, 0)
        if int(err) != 0:
            raise RuntimeError(f"cuLaunchKernel failed: {err}")


def current_stream() -> int:
    """Handle of PyTorch's current stream (the capture stream while recording a CUDA graph)."""
    return torch.cuda.current_stream().cuda_stream


# ---------------------------------------------------------------- convenience wrappers

WARPS_PER_BLOCK = 8


def gemv(mod: Module, W: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Launch:
    M, K = W.shape
    if K % 8:
        raise ValueError("K must be a multiple of 8")
    blocks = (M + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK
    return mod.launcher("gemv_f16", (blocks,), (32 * WARPS_PER_BLOCK,),
                        [W, x, y, ctypes.c_int(M), ctypes.c_int(K)])


def rmsnorm(mod: Module, x, w, out, eps: float = 1e-6) -> Launch:
    return mod.launcher("rmsnorm_f16", (1,), (1024,), [x, w, out, ctypes.c_int(x.numel()), ctypes.c_float(eps)])


def silu_mul(mod: Module, gate, up, out) -> Launch:
    n = gate.numel()
    return mod.launcher("silu_mul_f16", ((n + 255) // 256,), (256,), [gate, up, out, ctypes.c_int(n)])


def add_inplace(mod: Module, x, y) -> Launch:
    n = x.numel()
    return mod.launcher("add_inplace_f16", ((n + 255) // 256,), (256,), [x, y, ctypes.c_int(n)])


def read_sum(mod: Module, buf: torch.Tensor, out: torch.Tensor, blocks: int) -> Launch:
    n16 = buf.numel() * buf.element_size() // 16
    return mod.launcher("read_sum", (blocks,), (256,), [buf, ctypes.c_ulonglong(n16), out])
