"""Experiment 2: how close a hand-written FP16 GEMV gets to the memory-bandwidth ceiling.

For y = W x at batch 1, every weight byte is read once, so the floor is
    time >= (M*K*2 + K*2 + M*2) bytes / bandwidth.
Three things are measured on the same bytes:
  read_sum   a pure streaming read: the bandwidth this card delivers in practice
  ours       the GEMV in lab/kernels.cu
  torch.mv   PyTorch's matrix-vector product (cuBLAS)

L2 is 6 MB on an RTX 3090, so small matrices would partly be served from cache on repeated
calls. Each measurement rotates through enough copies that the working set is at least
256 MB, so every call reads from DRAM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import cuda_util as cu  # noqa: E402
from lab.timing import device_info, gpu_state, interleaved_gpu, summary  # noqa: E402

PEAK_GBPS = 936.0          # RTX 3090 spec: 19.5 Gbps GDDR6X x 384-bit
MIN_WORKING_SET = 256 << 20
SHAPES = [(4096, 4096), (11008, 4096), (4096, 11008), (2048, 2048), (5632, 2048), (2048, 5632),
          (16384, 4096), (32768, 4096), (65536, 4096)]
CALLS, TRIALS = 50, 15


def rotating(make_call, copies: int):
    calls = [make_call(i) for i in range(copies)]
    state = {"i": 0}

    def step():
        calls[state["i"]]()
        state["i"] = (state["i"] + 1) % copies

    return step


def main(out_path: Path) -> dict:
    mod = cu.Module()
    stream = cu.current_stream()
    blocks = torch.cuda.get_device_properties(0).multi_processor_count * 8
    rows = []
    for M, K in SHAPES:
        nbytes = M * K * 2 + K * 2 + M * 2
        copies = max(1, -(-MIN_WORKING_SET // (M * K * 2)))
        Ws = [torch.randn(M, K, device="cuda").half() for _ in range(copies)]
        x = torch.randn(K, device="cuda").half()
        ys = [torch.empty(M, device="cuda", dtype=torch.half) for _ in range(copies)]
        partial = torch.empty(blocks, device="cuda")

        ours = [cu.gemv(mod, Ws[i], x, ys[i]) for i in range(copies)]
        reads = [cu.read_sum(mod, Ws[i], partial, blocks) for i in range(copies)]
        t = interleaved_gpu({
            "read_sum": rotating(lambda i: (lambda: reads[i](stream)), copies),
            "ours": rotating(lambda i: (lambda: ours[i](stream)), copies),
            "torch_mv": rotating(lambda i: (lambda: torch.mv(Ws[i], x, out=ys[i])), copies),
        }, CALLS, TRIALS)

        floor_us = nbytes / (PEAK_GBPS * 1e9) * 1e6
        row = {"M": M, "K": K, "MB": round(nbytes / 1e6, 1), "copies": copies, "floor_us": round(floor_us, 2),
               "gpu_state": gpu_state()}
        for name in ("read_sum", "ours", "torch_mv"):
            s = summary(t[name], 1e6)
            # median = typical; best = least disturbed by other work on the GPU (noise only adds time)
            row[name] = {"us": s,
                         "gbps": round(nbytes / (s["median"] / 1e6) / 1e9, 1),
                         "pct_of_peak": round(100 * floor_us / s["median"], 1),
                         "best_pct_of_peak": round(100 * floor_us / s["min"], 1)}
        rows.append(row)
        print(f"{M:6d} x {K:5d}  {row['MB']:7.1f} MB  floor {floor_us:7.1f} us | "
              + " | ".join(f"{n} {row[n]['us']['median']:7.1f} us {row[n]['pct_of_peak']:5.1f}% (best {row[n]['best_pct_of_peak']:5.1f}%)"
                           for n in ("read_sum", "ours", "torch_mv")), flush=True)
        del Ws, ys, ours, reads
        torch.cuda.empty_cache()

    results = {"device": device_info(), "peak_gbps": PEAK_GBPS, "min_working_set_mb": MIN_WORKING_SET >> 20,
               "calls": CALLS, "trials": TRIALS, "rows": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/gemv_roofline.json"))
