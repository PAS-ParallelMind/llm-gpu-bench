"""Kernel operation benchmarks: gemm, attn, moe, allreduce (each a sweep + roofline).

Shared timing (timing.py) and the predictor (predict.py) live at the repo root; these modules
import `timing` from there. Driven by run.py (--bench) / validate_predict.py.
"""
