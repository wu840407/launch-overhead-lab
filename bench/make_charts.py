"""Charts for the README, from the JSON files in results/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "charts"
COLORS = {"read_sum": "#7f8c8d", "ours": "#1f6f8b", "torch_mv": "#d08c3a"}
LABELS = {"read_sum": "pure streaming read", "ours": "hand-written GEMV", "torch_mv": "torch.mv"}


def gemv_chart():
    runs = [json.loads(p.read_text()) for p in sorted(RES.glob("gemv_roofline_run*.json"))]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    # Points, not lines: two shapes can read the same number of bytes (M x K and K x M).
    offsets = {"read_sum": 0.97, "ours": 1.0, "torch_mv": 1.03}
    for name in ("read_sum", "ours", "torch_mv"):
        for i, run in enumerate(runs):
            rows = run["rows"]
            ax.scatter([r["MB"] * offsets[name] for r in rows], [r[name]["best_pct_of_peak"] for r in rows],
                       s=26, color=COLORS[name], alpha=1.0 if i == 0 else 0.4,
                       label=LABELS[name] if i == 0 else None, zorder=3)
    ax.axhline(100, ls="--", lw=1, color="#444")
    ax.text(9, 101.5, "spec peak 936 GB/s", fontsize=8, color="#444")
    ax.set_xscale("log")
    ax.set_xlabel("bytes read per call (MB, log scale)")
    ax.set_ylabel("% of peak bandwidth (best of 15)")
    ax.set_ylim(30, 108)
    ax.set_title("FP16 GEMV at batch 1 on RTX 3090 (9 shapes, two runs; faint = second run)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "gemv_bandwidth.png")


def step_chart():
    runs = [json.loads(p.read_text()) for p in sorted(RES.glob("decode_step_run*.json"))]
    configs = list(runs[0]["configs"])
    fig, axes = plt.subplots(1, len(configs), figsize=(9, 4), dpi=150)
    for ax, cfg in zip(axes, configs):
        c0 = runs[0]["configs"][cfg]
        bars = [("hand-written\neager", "driver", "eager_ms"), ("hand-written\ngraph", "driver", "graph_ms"),
                ("PyTorch ops\neager", "torch", "eager_ms"), ("PyTorch ops\ngraph", "torch", "graph_ms")]
        xs = range(len(bars))
        vals = [c0[k][m]["median"] for _, k, m in bars]
        cols = ["#1f6f8b", "#8fc1d4", "#d08c3a", "#f0c48a"]
        ax.bar(xs, vals, color=cols, width=0.65)
        for run in runs[1:]:
            c = run["configs"][cfg]
            ax.scatter(xs, [c[k][m]["median"] for _, k, m in bars], color="#222", s=12, zorder=3)
        for x, (_, k, m) in zip(xs, bars):
            if m == "eager_ms":
                ax.scatter([x], [c0[k]["issue_ms"]["median"]], marker="_", s=400, color="#c0392b", zorder=4)
        ax.axhline(c0["floor_ms"], ls="--", lw=1, color="#444")
        ax.text(3.35, c0["floor_ms"], " memory\n floor", fontsize=7, va="center", color="#444")
        ax.set_xticks(list(xs), [b[0] for b in bars], fontsize=7.5)
        ax.set_title(f"{cfg}: {c0['weight_gb']} GB of weights", fontsize=9)
        ax.set_ylabel("ms per step")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(-0.6, 4.2)
    fig.suptitle("Synthetic decode step: bars = run 1, dots = run 2, red tick = CPU time to issue the step",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "decode_step.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    gemv_chart()
    step_chart()
    print("charts written to", OUT)
    sys.exit(0)
