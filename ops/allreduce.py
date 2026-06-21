"""All-reduce bandwidth sweep across world sizes (NCCL) — the TP collective for serving.

After each transformer layer's attention and MLP, tensor-parallel serving all-reduces the
[tokens, hidden] activation across the TP ranks; that collective is pure overhead on the critical
path, so predicting serving latency needs its cost as a function of (message **bytes**, world
size). The cost is set by bytes moved and the interconnect, not the dtype -- a 1 MiB all-reduce
takes the same time whether it's bf16 or fp32 -- so this measures **achieved bandwidth vs message
size**, keyed on bytes (the bench is just `allreduce`, not per-dtype).

Detects the GPUs on the node and benchmarks NCCL all-reduce for world sizes 2, 4, 8, ... up to the
GPU count (W=1 has no communication, so it's skipped), over a sweep of message sizes. Each world
size runs as its own set of one-process-per-
GPU NCCL ranks (torch.multiprocessing.spawn; rank r -> cuda:r).

A batch of all-reduces is captured into a CUDA graph and we time graph replays -- exactly how vLLM
runs the decode TP collective, so the per-collective time is launch-free (no host launch / event /
dispatch overhead), the faithful serving number. Batching several collectives per graph amortizes
even the per-replay graph launch (one replay runs them back-to-back), which matters in the
latency-bound regime. (NCCL is warmed up on a side stream first, as graph capture of a cold
collective fails.) A barrier precedes timing and the latency is maxed across ranks. latency_ms is the prediction target (the predictor interpolates it directly in
log-bytes per world size, so no interconnect peak is needed); algbw = bytes/time and busbw =
algbw*2(W-1)/W (the algorithm-independent bus bandwidth) are kept as informational "achieved
GB/s" columns.

Single node only (device_count sees the local node); multi-node needs a torchrun launcher.

    python allreduce.py                          # auto-detect GPUs, sweep W=2,4,8,...
"""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Message sizes in BYTES: 4 KiB .. 256 MiB. Decode TP all-reduces ~[1, hidden] (KiB, latency-bound);
# prefill chunks ~[8192, hidden] (tens of MiB, bandwidth-bound).
SIZES = [4096, 16384, 65536, 262144, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28]
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_GRAPH_BATCH = 16   # all-reduces captured per CUDA graph; one replay runs them back-to-back so the
                    # graph launch is amortized over the batch (iters/_GRAPH_BATCH replays are timed).


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def world_sizes_for(n_gpus: int) -> list[int]:
    """2, 4, 8, ... up to n_gpus (and n_gpus itself if it isn't a power of two). World size 1 is
    dropped -- it has no communication; all-reduce needs >= 2 GPUs (empty list on a 1-GPU node)."""
    ws, w = [], 2
    while w <= n_gpus:
        ws.append(w)
        w *= 2
    if n_gpus >= 2 and n_gpus not in ws:
        ws.append(n_gpus)
    return ws


def _worker(rank: int, world_size: int, byte_sizes: list[int], dtype_name: str,
            iters: int, warmup: int, port: int, out_list) -> None:
    """One NCCL rank (rank r -> cuda:r): sweep message sizes, time all-reduce, rank 0 collects."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=dev)
    dt = _DTYPES[dtype_name]
    try:
        for nbytes in byte_sizes:
            n_elem = max(1, nbytes // dt.itemsize)
            t = torch.ones(n_elem, dtype=dt, device=dev)

            # Capture the all-reduce into a CUDA graph and time replays -- this is how vLLM runs
            # the decode TP collective, so it's the faithful number (launch fully amortized). NCCL
            # needs the collective warmed up on a side stream first (to set up its buffers and
            # connections); capturing a cold collective fails.
            side = torch.cuda.Stream(dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    dist.all_reduce(t)
            torch.cuda.current_stream(dev).wait_stream(side)
            torch.cuda.synchronize(dev)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(_GRAPH_BATCH):   # a batch of collectives per graph, so one replay
                    dist.all_reduce(t)          # runs them back-to-back launch-free (as serving does)
            g.replay()                          # warm the replay path
            torch.cuda.synchronize(dev)
            dist.barrier()

            reps = max(1, round(iters / _GRAPH_BATCH))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):
                g.replay()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end) / (_GRAPH_BATCH * reps)

            lat = torch.tensor([ms], device=dev)
            dist.all_reduce(lat, op=dist.ReduceOp.MAX)   # collective latency = slowest rank
            if rank == 0:
                out_list.append((n_elem * dt.itemsize, lat.item()))
            del g, t
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


def _run_world(world_size: int, sizes: list[int], dtype: str, iters: int, warmup: int) -> list:
    """Spawn `world_size` NCCL ranks, return rank 0's [(bytes, latency_ms), ...]."""
    out = mp.Manager().list()
    mp.spawn(_worker, args=(world_size, sizes, dtype, iters, warmup, _free_port(), out),
             nprocs=world_size, join=True)
    return list(out)


def _record(world_size: int, nbytes: int, ms: float) -> dict:
    """latency_ms is the prediction target (interpolated directly); algbw/busbw are informational
    (achieved GB/s -- busbw = algbw * 2(W-1)/W is the algorithm-independent bus bandwidth)."""
    sec = ms * 1e-3
    algbw = nbytes / sec / 1e9 if sec > 0 else 0.0               # GB/s the tensor sees
    busfac = 2.0 * (world_size - 1) / world_size                # ring all-reduce bus factor (0 at W=1)
    return {"shape": {"world_size": world_size, "bytes": nbytes}, "latency_ms": ms,
            "algbw_gbps": algbw, "busbw_gbps": algbw * busfac}


def run_full_allreduce_sweep(*, sizes: list[int] | None = None, dtype: str = "bf16",
                             iters: int = 50, warmup: int = 20,
                             max_gpus: int | None = None, verbose: bool = True):
    """Detect GPUs and sweep NCCL all-reduce over world sizes 2,4,8,...×N and message sizes.
    Returns (n_gpus, world_sizes, results) where results is a list of per-(W, bytes) dicts."""
    sizes = sizes or SIZES
    n_gpus = torch.cuda.device_count()
    if max_gpus:
        n_gpus = min(n_gpus, max_gpus)
    ws_list = world_sizes_for(n_gpus)
    if not ws_list:
        raise SystemExit(f"all-reduce needs >= 2 GPUs (found {n_gpus}).")
    results: list[dict] = []
    for W in ws_list:
        if verbose:
            print(f"\n== world_size {W} ==")
            print(f"  {'bytes':>12} {'latency_ms':>11} {'algbw_GB/s':>11} {'busbw_GB/s':>11}")
        for nbytes, ms in _run_world(W, sizes, dtype, iters, warmup):
            r = _record(W, nbytes, ms)
            results.append(r)
            if verbose:
                print(f"  {nbytes:>12} {ms:>11.4f} {r['algbw_gbps']:>11.1f} {r['busbw_gbps']:>11.1f}")
    return n_gpus, ws_list, results


def write_results(out: str, gpu: str, n_gpus: int, results: list[dict]) -> None:
    """Write the unified result JSON (hardware carries n_gpus; the predictor interpolates
    latency_ms directly, so there is no bus/compute peak to record)."""
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({
        "hardware": {"gpu": gpu, "n_gpus": n_gpus},
        "operation": {"bench": "allreduce", "impl": {
            "torch": torch.__version__,
            "nccl": ".".join(map(str, torch.cuda.nccl.version()))}},
        "results": results,
    }, indent=2))
    print(f"\nwrote {out_p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", default="bf16", choices=list(_DTYPES),
                    help="allocation dtype (cost is byte-driven; this is just the measurement vehicle).")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-gpus", type=int, default=None, help="cap the world size (default: all).")
    ap.add_argument("--sizes", type=int, nargs="+", default=None, help="message sizes in bytes.")
    ap.add_argument("--out", type=str, default="results/allreduce.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — all-reduce benchmark needs GPUs.")
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu} x{torch.cuda.device_count()} | NCCL all-reduce (achieved bandwidth vs bytes)")
    n_gpus, ws_list, results = run_full_allreduce_sweep(
        sizes=args.sizes, dtype=args.dtype, iters=args.iters, warmup=args.warmup,
        max_gpus=args.max_gpus)
    write_results(args.out, gpu, n_gpus, results)


if __name__ == "__main__":
    main()
