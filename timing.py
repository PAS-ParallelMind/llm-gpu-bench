"""GPU kernel timing — CUDA events, warmup, per-iteration L2 flush, robust stats.

Memory-bound kernels (decode GEMM, KV reads) are acutely sensitive to a hot L2,
so we overwrite a buffer larger than the device L2 between timed runs. Modeled on
Triton's ``do_bench`` but returns full percentiles: median is the number the perf
model should consume (typical), min is the clean-room peak (sanity vs roofline).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class Timing:
    median_ms: float
    min_ms: float
    mean_ms: float
    p10_ms: float
    p90_ms: float
    std_ms: float
    n: int
    samples_ms: list[float] = field(repr=False, default_factory=list)

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> "Timing":
        s = sorted(samples_ms)
        n = len(s)

        def pct(q: float) -> float:
            if n == 1:
                return s[0]
            pos = q * (n - 1)
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            return s[lo] + (s[hi] - s[lo]) * (pos - lo)

        return cls(
            median_ms=pct(0.50),
            min_ms=s[0],
            mean_ms=statistics.fmean(s),
            p10_ms=pct(0.10),
            p90_ms=pct(0.90),
            std_ms=statistics.pstdev(s) if n > 1 else 0.0,
            n=n,
            samples_ms=samples_ms,
        )


def _l2_flush_buffer(device: torch.device) -> torch.Tensor:
    """A buffer big enough to evict the device L2 when zeroed (2x L2, floor 128 MiB).

    Ada's L2 is 72 MiB on the 4090, so the floor has to clear it comfortably."""
    props = torch.cuda.get_device_properties(device)
    l2_bytes = getattr(props, "L2_cache_size", 0) or 0
    nbytes = max(2 * l2_bytes, 128 * 1024 * 1024)
    return torch.empty(nbytes, dtype=torch.int8, device=device)


def measure(
    fn: Callable[[], object],
    *,
    warmup: int = 25,
    iters: int = 100,
    flush_l2: bool = True,
    device: torch.device | int | None = None,
) -> Timing:
    """Time a zero-arg kernel closure.

    All tensor allocation must happen *outside* ``fn`` — the closure should
    launch only the kernel under test. The per-iteration L2 flush is enqueued
    before ``start`` records, so it is not counted in the timed region.
    """
    if device is None:
        device = torch.cuda.current_device()
    if not isinstance(device, torch.device):
        device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)

    cache = _l2_flush_buffer(device) if flush_l2 else None

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        if cache is not None:
            cache.zero_()          # evict L2; enqueued before `start`, so untimed
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return Timing.from_samples(samples)
