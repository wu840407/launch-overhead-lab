"""Experiment 3: a synthetic decode step, eager versus CUDA graph, written two ways.

Each layer runs the kernels of a Llama-style block at batch 1:
    rmsnorm, q, k, v, o projections, residual add,
    rmsnorm, gate, up, silu*mul, down projection, residual add.
Attention over the KV cache is left out on purpose: at short context it is small next to the
weight reads, and leaving it out keeps the step a function of weight bytes and kernel count.
This models the mechanism, not a real model.

The same step is issued two ways, on the same weights:
  driver   12 hand-written kernels per layer, each one cuLaunchKernel with pre-packed arguments
  torch    the same math written as ordinary PyTorch ops (torch.mv, elementwise ops), the way
           eager-mode model code is written; more kernels per layer and more CPU work per kernel

For each, three numbers per step:
  issue    host time to enqueue the whole step (no synchronize)
  eager    host time per step when steps run back to back
  graph    the same step recorded once into a CUDA graph and replayed

Shapes are public architectures, used only for their sizes:
    1B-class   hidden 2048, intermediate 5632, 22 layers  (TinyLlama-1.1B)
    7B-class   hidden 4096, intermediate 11008, 32 layers (Llama-2-7B)
Weights are random FP16 (scaled by 0.02).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import cuda_util as cu  # noqa: E402
from lab.timing import device_info, gpu_state, interleaved_wall, summary  # noqa: E402

PEAK_GBPS = 936.0
CONFIGS = {"1B-class": (2048, 5632, 22), "7B-class": (4096, 11008, 32)}
STEPS, TRIALS = 20, 15
EPS = 1e-6


def build_weights(hidden: int, inter: int, layers: int):
    def h(*shape):
        return torch.randn(*shape, device="cuda").half() * 0.02

    weights = []
    for _ in range(layers):
        weights.append({
            "q": h(hidden, hidden), "k": h(hidden, hidden), "v": h(hidden, hidden), "o": h(hidden, hidden),
            "gate": h(inter, hidden), "up": h(inter, hidden), "down": h(hidden, inter),
            "n1": torch.ones(hidden, device="cuda").half(), "n2": torch.ones(hidden, device="cuda").half(),
        })
    nbytes = sum(t.numel() * t.element_size() for layer in weights for t in layer.values())
    return weights, nbytes


def driver_step(mod, weights, x, hidden: int, inter: int):
    b = {n: torch.empty(hidden, device="cuda", dtype=torch.half) for n in ("norm", "q", "k", "v", "o", "down")}
    gate, up, act = (torch.empty(inter, device="cuda", dtype=torch.half) for _ in range(3))
    launches = []
    for w in weights:
        launches += [
            cu.rmsnorm(mod, x, w["n1"], b["norm"], EPS),
            cu.gemv(mod, w["q"], b["norm"], b["q"]),
            cu.gemv(mod, w["k"], b["norm"], b["k"]),
            cu.gemv(mod, w["v"], b["norm"], b["v"]),
            cu.gemv(mod, w["o"], b["v"], b["o"]),      # stands in for o_proj(attention output)
            cu.add_inplace(mod, x, b["o"]),
            cu.rmsnorm(mod, x, w["n2"], b["norm"], EPS),
            cu.gemv(mod, w["gate"], b["norm"], gate),
            cu.gemv(mod, w["up"], b["norm"], up),
            cu.silu_mul(mod, gate, up, act),
            cu.gemv(mod, w["down"], act, b["down"]),
            cu.add_inplace(mod, x, b["down"]),
        ]

    def step():
        stream = cu.current_stream()
        for launch in launches:
            launch(stream)

    return step, len(launches)


def torch_step(weights, x):
    def rmsnorm(v, w):
        vf = v.float()
        return (vf * torch.rsqrt(vf.pow(2).mean() + EPS)).half() * w

    def step():
        for w in weights:
            n = rmsnorm(x, w["n1"])
            torch.mv(w["q"], n)
            torch.mv(w["k"], n)
            v = torch.mv(w["v"], n)
            x.add_(torch.mv(w["o"], v))
            n = rmsnorm(x, w["n2"])
            act = F.silu(torch.mv(w["gate"], n)) * torch.mv(w["up"], n)
            x.add_(torch.mv(w["down"], act))

    return step


def count_kernels(step) -> int:
    from torch.profiler import ProfilerActivity, profile

    step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def issue_time(step, trials: int) -> list[float]:
    out = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        out.append(time.perf_counter() - t0)
        torch.cuda.synchronize()
    return out


def measure(step, n_kernels: int, floor_ms: float) -> dict:
    graph = torch.cuda.CUDAGraph()
    step()                       # warm up allocator and kernels before capture
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        step()
    before = gpu_state()
    t = interleaved_wall({"eager": step, "graph": graph.replay}, STEPS, TRIALS)
    r = {
        "kernels_per_step": n_kernels,
        "issue_ms": summary(issue_time(step, TRIALS), 1e3),
        "eager_ms": summary(t["eager"], 1e3),
        "graph_ms": summary(t["graph"], 1e3),
        "gpu_state": {"before": before, "after": gpu_state()},
    }
    e, g = r["eager_ms"]["median"], r["graph_ms"]["median"]
    r["eager_pct_of_floor"] = round(100 * floor_ms / e, 1)
    r["graph_pct_of_floor"] = round(100 * floor_ms / g, 1)
    r["eager_over_graph"] = round(e / g, 2)
    r["eager_minus_graph_us_per_kernel"] = round((e - g) * 1e3 / n_kernels, 2)
    del graph
    return r


def main(out_path: Path) -> dict:
    mod = cu.Module()
    results = {"device": device_info(), "peak_gbps": PEAK_GBPS, "steps": STEPS, "trials": TRIALS, "configs": {}}
    for name, (hidden, inter, layers) in CONFIGS.items():
        weights, nbytes = build_weights(hidden, inter, layers)
        floor_ms = nbytes / (PEAK_GBPS * 1e9) * 1e3
        x = torch.randn(hidden, device="cuda").half()
        d_step, d_kernels = driver_step(mod, weights, x, hidden, inter)
        t_step = torch_step(weights, x)
        cfg = {"hidden": hidden, "intermediate": inter, "layers": layers,
               "weight_gb": round(nbytes / 1e9, 3), "floor_ms": round(floor_ms, 3),
               "driver": measure(d_step, d_kernels, floor_ms),
               "torch": measure(t_step, count_kernels(t_step), floor_ms)}
        results["configs"][name] = cfg
        for kind in ("driver", "torch"):
            r = cfg[kind]
            print(f"{name:8s} {kind:6s} {r['kernels_per_step']:4d} kernels | floor {floor_ms:6.2f} ms | "
                  f"issue {r['issue_ms']['median']:6.2f} | eager {r['eager_ms']['median']:6.2f} "
                  f"({r['eager_pct_of_floor']:5.1f}%) | graph {r['graph_ms']['median']:6.2f} "
                  f"({r['graph_pct_of_floor']:5.1f}%) | eager/graph {r['eager_over_graph']:.2f}", flush=True)
        del weights, d_step, t_step, x
        torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/decode_step.json"))
