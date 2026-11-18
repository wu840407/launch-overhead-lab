"""Why did the same graph replay take 16.6 ms in one run and 19.6-20.2 ms in another?

Replays the 7B-class driver step and one 90 MB GEMV in alternation for a few minutes,
sampling clocks, temperature and power around every measurement. If the step time follows
the memory clock or the power state, the difference is the card, not the code.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import cuda_util as cu  # noqa: E402
from lab.timing import gpu_per_call, gpu_state, wall_per_call  # noqa: E402
from decode_step import build_weights, driver_step  # noqa: E402

ROUNDS = 12


def main(out_path: Path):
    mod = cu.Module()
    hidden, inter, layers = 4096, 11008, 32
    weights, nbytes = build_weights(hidden, inter, layers)
    x = torch.randn(hidden, device="cuda").half()
    step, n = driver_step(mod, weights, x, hidden, inter)
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    W = weights[0]["gate"]                     # 11008 x 4096, 90 MB
    xin = torch.randn(4096, device="cuda").half()
    y = torch.empty(11008, device="cuda", dtype=torch.half)
    gemv = cu.gemv(mod, W, xin, y)
    stream = cu.current_stream()

    rows = []
    t_start = time.time()
    for r in range(ROUNDS):
        before = gpu_state()
        step_ms = sorted(wall_per_call(graph.replay, 10, 3, warmup=1))[1] * 1e3
        mid = gpu_state()
        gemv_us = sorted(gpu_per_call(lambda: gemv(stream), 50, 3, warmup=1))[1] * 1e6
        after = gpu_state()
        row = {"t_s": round(time.time() - t_start, 1), "step_ms": round(step_ms, 2), "gemv_us": round(gemv_us, 1),
               "gemv_gbps": round(W.numel() * 2 / (gemv_us / 1e6) / 1e9, 1),
               "before": before, "mid": mid, "after": after}
        rows.append(row)
        print(f"t={row['t_s']:6.1f}s  step {row['step_ms']:6.2f} ms  gemv {row['gemv_us']:6.1f} us "
              f"({row['gemv_gbps']} GB/s) | sm {mid.get('sm_mhz')} MHz mem {mid.get('mem_mhz')} MHz "
              f"{mid.get('temp_c')} C {mid.get('power_w')} W util {before.get('util_pct')}%", flush=True)
        time.sleep(5)
    out_path.write_text(json.dumps({"rows": rows, "kernels_per_step": n, "weight_gb": nbytes / 1e9}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/variance.json"))
