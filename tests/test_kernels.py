"""Correctness of the hand-written kernels against PyTorch (needs a CUDA GPU)."""
import ctypes

import pytest
import torch

from lab import cuda_util as cu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture(scope="module")
def mod():
    return cu.Module()


def run(launch):
    launch(cu.current_stream())
    torch.cuda.synchronize()


@pytest.mark.parametrize("M, K", [(1, 8), (7, 64), (33, 256), (4096, 4096), (11008, 4096), (4096, 11008), (1000, 5632)])
def test_gemv_matches_fp32_reference(mod, M, K):
    g = torch.Generator(device="cuda").manual_seed(M * 31 + K)
    W = torch.randn(M, K, device="cuda", generator=g).half()
    x = torch.randn(K, device="cuda", generator=g).half()
    y = torch.empty(M, device="cuda", dtype=torch.half)
    run(cu.gemv(mod, W, x, y))
    ref = W.float() @ x.float()
    # FP32 accumulation in a different order, then one rounding to FP16.
    tol = 2e-3 * ref.abs().max().item() + 1e-3
    assert (y.float() - ref).abs().max().item() <= tol


def test_gemv_rejects_unaligned_k(mod):
    W = torch.zeros(4, 12, device="cuda", dtype=torch.half)
    with pytest.raises(ValueError):
        cu.gemv(mod, W, torch.zeros(12, device="cuda", dtype=torch.half), torch.zeros(4, device="cuda", dtype=torch.half))


def test_gemv_detects_a_wrong_row(mod):
    # Guard against a test that cannot fail: corrupt one weight row and expect a mismatch.
    W = torch.randn(64, 256, device="cuda").half()
    x = torch.randn(256, device="cuda").half()
    y = torch.empty(64, device="cuda", dtype=torch.half)
    run(cu.gemv(mod, W, x, y))
    W2 = W.clone()
    W2[17] += 1.0
    assert (y.float() - (W2.float() @ x.float())).abs()[17].item() > 1.0


@pytest.mark.parametrize("n", [64, 2048, 4096, 5000])
def test_rmsnorm(mod, n):
    x = torch.randn(n, device="cuda").half()
    w = torch.randn(n, device="cuda").half()
    out = torch.empty_like(x)
    run(cu.rmsnorm(mod, x, w, out, eps=1e-6))
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean() + 1e-6) * w.float()
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=1e-2)


def test_silu_mul_and_add(mod):
    n = 11008
    g, u = torch.randn(n, device="cuda").half(), torch.randn(n, device="cuda").half()
    out = torch.empty_like(g)
    run(cu.silu_mul(mod, g, u, out))
    torch.testing.assert_close(out.float(), torch.nn.functional.silu(g.float()) * u.float(), atol=1e-2, rtol=1e-2)

    x, y = torch.randn(n, device="cuda").half(), torch.randn(n, device="cuda").half()
    ref = x.float() + y.float()
    run(cu.add_inplace(mod, x, y))
    torch.testing.assert_close(x.float(), ref, atol=1e-2, rtol=1e-2)


def test_read_sum_touches_every_byte(mod):
    buf = torch.zeros(1 << 20, device="cuda", dtype=torch.uint8)
    out = torch.zeros(64, device="cuda")
    run(cu.read_sum(mod, buf, out, blocks=64))
    base = out.sum().item()
    buf[-1] = 0x10  # change one byte in the last 16-byte word
    run(cu.read_sum(mod, buf, out, blocks=64))
    assert out.sum().item() != base


def test_launches_are_recorded_into_a_cuda_graph(mod):
    """Driver-API launches on PyTorch's capture stream end up in the graph."""
    x = torch.zeros(256, device="cuda").half()
    one = torch.ones(256, device="cuda").half()
    add = cu.add_inplace(mod, x, one)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stream = cu.current_stream()
        for _ in range(3):
            add(stream)
    x.zero_()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert x.float().eq(6.0).all()
