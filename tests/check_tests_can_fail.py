"""Break the kernels on purpose, one change at a time, and check that the tests fail.

A test suite that passes is only evidence if it would have failed on a wrong kernel.
Each mutation below edits a copy of the source, runs pytest against it, and restores the
original. Run from the repository root:

    python tests/check_tests_can_fail.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "lab" / "kernels.cu"
UTIL = ROOT / "lab" / "cuda_util.py"

MUTATIONS = [
    ("warp reduction skips the first step", KERNELS,
     "for (int off = 16; off > 0; off >>= 1) v +=", "for (int off = 8; off > 0; off >>= 1) v +="),
    ("GEMV drops the last 16-byte chunk of each row", KERNELS,
     "for (int i = lane; i < (K >> 3); i += 32) {", "for (int i = lane; i < (K >> 3) - 1; i += 32) {"),
    ("GEMV writes the result from the wrong lane", KERNELS,
     "if (lane == 0) y[row] = __float2half(acc);", "if (lane == 1) y[row] = __float2half(acc);"),
    ("GEMV maps two warps to one row", KERNELS,
     "const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;",
     "const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 6;"),
    ("block sum keeps only the first warp", KERNELS,
     "v = threadIdx.x < (blockDim.x >> 5) ? partial[threadIdx.x] : 0.f;",
     "v = threadIdx.x < 1 ? partial[threadIdx.x] : 0.f;"),
    ("RMSNorm divides by n - 1 instead of n", KERNELS,
     "scale = rsqrtf(s / n + eps);", "scale = rsqrtf(s / (n - 1) + eps);"),
    ("RMSNorm forgets the weight", KERNELS,
     "out[i] = __float2half(__half2float(x[i]) * scale * __half2float(w[i]));",
     "out[i] = __float2half(__half2float(x[i]) * scale);"),
    ("SiLU uses exp(+g)", KERNELS, "__expf(-g)", "__expf(g)"),
    ("residual add skips the last element", KERNELS,
     "if (i < n) x[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));",
     "if (i < n - 1) x[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));"),
    ("streaming read stops halfway", KERNELS, "i < n; i += stride", "i < n / 2; i += stride"),
    ("launches ignore the stream (would escape graph capture)", UTIL,
     "stream, self._params_ptr, 0)", "0, self._params_ptr, 0)"),
]


def main() -> int:
    missed = 0
    for name, path, old, new in MUTATIONS:
        src = path.read_text(encoding="utf-8")
        if src.count(old) != 1:
            print(f"SETUP ERROR  {name}: pattern found {src.count(old)} times")
            return 2
        path.write_text(src.replace(old, new, 1), encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
                               cwd=ROOT, capture_output=True, text=True)
        finally:
            path.write_text(src, encoding="utf-8")
        caught = r.returncode != 0
        missed += not caught
        first = next((l for l in r.stdout.splitlines() if l.startswith("FAILED")), "")
        print(f"{'caught' if caught else 'MISSED':6s}  {name}   {first[:90]}")
    print(f"\n{len(MUTATIONS) - missed} of {len(MUTATIONS)} deliberate bugs caught")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
