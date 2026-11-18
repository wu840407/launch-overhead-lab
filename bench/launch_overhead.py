"""Experiment 1: what one kernel launch costs, three ways.

  torch eager   a tiny in-place PyTorch op (what an eager-mode serving loop pays per op)
  driver eager  an empty kernel launched with cuLaunchKernel, arguments packed once
                (the cost of launching from Python with almost nothing else in the way)
  CUDA graph    the same launches recorded once and replayed (no per-launch CPU work)

The kernels do no work, so every number here is launch overhead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import cuda_util as cu  # noqa: E402
from lab.timing import device_info, gpu_per_call, gpu_state, summary, wall_per_call  # noqa: E402

CALLS = 20_000      # launches per eager trial
GRAPH_LEN = 1_000   # launches recorded into one graph
REPLAYS = 20        # graph replays per trial
TRIALS = 15


def graph_of(launch_one, n: int) -> torch.cuda.CUDAGraph:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            launch_one()
    return g


def main(out_path: Path) -> dict:
    mod = cu.Module()
    t = torch.zeros(1, device="cuda")
    empty = mod.launcher("empty_kernel", (1,), (32,), [])
    stream = cu.current_stream()

    def torch_op():
        t.add_(1.0)

    def driver_launch():
        empty(stream)

    results = {"device": device_info(), "gpu_state": gpu_state(), "calls": CALLS, "graph_len": GRAPH_LEN,
               "trials": TRIALS}
    results["torch_eager_us"] = summary(wall_per_call(torch_op, CALLS, TRIALS), 1e6)
    results["driver_eager_us"] = summary(wall_per_call(driver_launch, CALLS, TRIALS), 1e6)

    g_torch = graph_of(torch_op, GRAPH_LEN)

    def driver_on_capture_stream():
        empty(cu.current_stream())  # inside capture this is the capture stream

    g_driver = graph_of(driver_on_capture_stream, GRAPH_LEN)
    per_kernel = lambda s: [x / GRAPH_LEN for x in s]  # noqa: E731
    results["torch_graph_us"] = summary(per_kernel(wall_per_call(g_torch.replay, REPLAYS, TRIALS)), 1e6)
    results["driver_graph_us"] = summary(per_kernel(wall_per_call(g_driver.replay, REPLAYS, TRIALS)), 1e6)
    results["driver_graph_gpu_us"] = summary(per_kernel(gpu_per_call(g_driver.replay, REPLAYS, TRIALS)), 1e6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/launch_overhead.json")
    r = main(out)
    print(json.dumps(r["device"]))
    for k in ("torch_eager_us", "driver_eager_us", "torch_graph_us", "driver_graph_us", "driver_graph_gpu_us"):
        s = r[k]
        print(f"{k:22s} median {s['median']:7.2f} us   (min {s['min']:.2f}, max {s['max']:.2f}, n={s['n']})")
