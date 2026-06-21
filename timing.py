"""GPU kernel timing — CUPTI on-device timestamps, L2 flush, robust stats.

We time each kernel from the GPU's *own* per-kernel start/end timestamps (CUPTI Activity API),
not from host-side CUDA events. That excludes host launch latency *and* the fixed CUDA-event +
grid-dispatch overhead (~1us) -- negligible for big kernels but ~10% of a few-microsecond decode
GEMM, so device timestamps are the honest primitive for the small-op corner (and match vLLM's
CUDA-graph decode, which amortizes launch away). `measure` sums the durations of every kernel an
op launches (a fused MoE fires several) per iteration; median is the number the perf model uses.

Memory-bound kernels (decode GEMM, KV reads) are acutely sensitive to a hot L2, so a buffer
larger than the device L2 is zeroed before each call (cold HBM, as serving sees it). That flush
is a kernel too; it is excluded from the op sum by its exact CUPTI name (a captured-once int8
fill our bf16/fp4 ops never emit) and doubles as the per-iteration delimiter.

Requires NVIDIA's `cupti-python` (version-matched to the CUDA toolkit; Linux only) -- there is no
CUDA-event fallback, on purpose: accuracy over portability.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable

import torch

from cupti import cupti as _cupti

_CK = _cupti.ActivityKind.CONCURRENT_KERNEL


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


def _resolve_device(device) -> torch.device:
    if device is None:
        device = torch.cuda.current_device()
    if not isinstance(device, torch.device):
        device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    return device


def _l2_flush_buffer(device: torch.device) -> torch.Tensor:
    """A buffer big enough to evict the device L2 when zeroed (2x L2, floor 128 MiB).

    Ada's L2 is 72 MiB on the 4090, so the floor has to clear it comfortably."""
    props = torch.cuda.get_device_properties(device)
    l2_bytes = getattr(props, "L2_cache_size", 0) or 0
    nbytes = max(2 * l2_bytes, 128 * 1024 * 1024)
    return torch.empty(nbytes, dtype=torch.int8, device=device)


_FLUSH_NAME: str | None = None          # captured once: the int8 fill kernel of the L2 flush


def _flush_kernel_name(device: torch.device) -> str | None:
    """The exact CUPTI kernel name of the L2 flush (`int8.zero_()`), captured once and cached, so
    it can be excluded from an op's kernel sum. Our bf16/fp4 ops never emit an int8 fill."""
    global _FLUSH_NAME
    if _FLUSH_NAME is not None:
        return _FLUSH_NAME
    buf = torch.empty(4 * 1024 * 1024, dtype=torch.int8, device=device)   # same dtype as the flush
    buf.zero_()                         # compile the fill kernel first
    torch.cuda.synchronize(device)
    names: list[str] = []

    def req():
        return 1 * 1024 * 1024, 0

    def comp(activities):
        names.extend(a.name for a in activities if a.kind == _CK)

    _cupti.activity_register_callbacks(req, comp)
    _cupti.activity_enable(_CK)
    try:
        buf.zero_()
        torch.cuda.synchronize(device)
        _cupti.activity_flush_all(1)
    finally:
        _cupti.activity_disable(_CK)
    _FLUSH_NAME = names[-1] if names else None
    return _FLUSH_NAME


def _per_iter_samples(records: list, iters: int, flush_l2: bool, flush_name: str | None) -> list[float]:
    """Per-iteration op-kernel time (ms) from (start, end, name) records. With a flush each
    iteration is delimited by its flush kernel; otherwise op kernels are split into `iters` chunks."""
    records.sort(key=lambda r: r[0])    # by device start time -> launch order (single stream)
    if flush_l2 and flush_name is not None:
        samples, cur, started = [], 0.0, False
        for start, end, name in records:
            if name == flush_name:      # flush marks the boundary; its own time is excluded
                if started:
                    samples.append(cur)
                cur, started = 0.0, True
            elif started:
                cur += (end - start) / 1e6
        if started:
            samples.append(cur)
        return samples
    op = [(e - s) / 1e6 for s, e, name in records if name != flush_name]
    if not op:
        return op
    k = max(1, round(len(op) / iters))  # kernels per op call
    return [sum(op[i:i + k]) for i in range(0, len(op), k)]


def measure(
    fn: Callable[[], object],
    *,
    warmup: int = 25,
    iters: int = 100,
    flush_l2: bool = True,
    device: torch.device | int | None = None,
) -> Timing:
    """Time a zero-arg kernel closure via CUPTI on-device timestamps (all allocation outside `fn`;
    it should launch only the kernel under test). Sums each call's CONCURRENT_KERNEL durations --
    launch latency and event/dispatch overhead excluded -- grouped per iteration. With `flush_l2`
    a >L2 buffer is zeroed before each call (cold HBM) and excluded from the op sum by name.
    """
    device = _resolve_device(device)
    cache = _l2_flush_buffer(device) if flush_l2 else None
    flush_name = _flush_kernel_name(device) if cache is not None else None
    if cache is not None and flush_name is None:
        raise RuntimeError("could not identify the L2-flush kernel for CUPTI exclusion")

    def step():
        if cache is not None:
            cache.zero_()               # evict L2 -> fn() reads cold (excluded from the op sum)
        fn()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)

    records: list[tuple[int, int, str]] = []

    def req():
        return 16 * 1024 * 1024, 0      # (buffer bytes, max records; 0 = let CUPTI fill it)

    def comp(activities):
        records.extend((a.start, a.end, a.name) for a in activities if a.kind == _CK)

    _cupti.activity_register_callbacks(req, comp)
    _cupti.activity_enable(_CK)
    try:
        for _ in range(iters):
            step()
        torch.cuda.synchronize(device)
        _cupti.activity_flush_all(1)    # forced flush -> comp delivers all records
    finally:
        _cupti.activity_disable(_CK)

    samples = _per_iter_samples(records, iters, flush_l2, flush_name)
    if not samples:
        raise RuntimeError("CUPTI captured no op CONCURRENT_KERNEL activities")
    return Timing.from_samples(samples)


def progress(total: int, desc: str):
    """A tqdm progress bar over `total` sweep configs; a no-op if tqdm is absent."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="cfg")
    except ImportError:
        class _Null:
            def update(self, n: int = 1):
                pass

            def set_postfix_str(self, s: str):
                pass

            def close(self):
                pass

        return _Null()
