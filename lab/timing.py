"""Timing helpers. Every result is a list of repeated trials; reports use the median and the
spread (min, max) so run-to-run noise is visible next to every number."""
from __future__ import annotations

import statistics
import time

import torch


def wall_per_call(fn, calls: int, trials: int, warmup: int = 2) -> list[float]:
    """Seconds per call measured on the host, including the final synchronize.

    For an eager loop this is what the application waits for: the CPU issue cost and the
    GPU execution overlap, and whichever is slower sets the rate."""
    out = []
    for t in range(warmup + trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
        if t >= warmup:
            out.append((time.perf_counter() - t0) / calls)
    return out


def interleaved_wall(fns: dict, calls: int, trials: int, warmup: int = 2) -> dict[str, list[float]]:
    """wall_per_call for several variants, run A-B-B-A so slow drift and interference from
    other work on the GPU hit every variant equally."""
    out = {name: [] for name in fns}
    items = list(fns.items())
    for t in range(warmup + trials):
        for name, fn in (items if t % 2 == 0 else items[::-1]):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(calls):
                fn()
            torch.cuda.synchronize()
            if t >= warmup:
                out[name].append((time.perf_counter() - t0) / calls)
    return out


def interleaved_gpu(fns: dict, calls: int, trials: int, warmup: int = 2) -> dict[str, list[float]]:
    """gpu_per_call for several variants, A-B-C then C-B-A, so interference from other work
    on a shared GPU is spread over every variant instead of landing on one of them."""
    out = {name: [] for name in fns}
    items = list(fns.items())
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for t in range(warmup + trials):
        for name, fn in (items if t % 2 == 0 else items[::-1]):
            torch.cuda.synchronize()
            start.record()
            for _ in range(calls):
                fn()
            end.record()
            end.synchronize()
            if t >= warmup:
                out[name].append(start.elapsed_time(end) / 1e3 / calls)
    return out


def gpu_per_call(fn, calls: int, trials: int, warmup: int = 2) -> list[float]:
    """Seconds per call measured with CUDA events (GPU timeline)."""
    out = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for t in range(warmup + trials):
        torch.cuda.synchronize()
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        end.synchronize()
        if t >= warmup:
            out.append(start.elapsed_time(end) / 1e3 / calls)
    return out


def summary(samples: list[float], scale: float = 1.0) -> dict:
    s = sorted(x * scale for x in samples)
    return {"median": statistics.median(s), "min": s[0], "max": s[-1], "n": len(s)}


def device_info() -> dict:
    import platform

    p = torch.cuda.get_device_properties(0)
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {
        "gpu": p.name,
        "sm": f"{p.major}.{p.minor}",
        "sms": p.multi_processor_count,
        "cpu": cpu,
        "wsl2": "microsoft" in platform.release().lower(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def gpu_state() -> dict:
    """Clocks, temperature and power right now. GeForce cards cannot lock clocks, so these are
    recorded next to every measurement to explain run-to-run differences."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        state = {
            "sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "mem_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
            "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "power_w": round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1),
            "util_pct": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
            "used_mib": pynvml.nvmlDeviceGetMemoryInfo(h).used >> 20,
        }
        pynvml.nvmlShutdown()
        return state
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__}
