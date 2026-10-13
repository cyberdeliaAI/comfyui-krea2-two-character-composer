"""Synthetic adapter benchmark; not a Krea 2 image-generation benchmark.

Runs the old dense formula and the optimized layer on identical tensors/masks.
Examples: python tools/benchmark_lora.py --device cuda --dtype float16
          python tools/benchmark_lora.py --device cpu --dtype float32
The base projection is a zero tensor to measure adapter overhead separately.
"""
import argparse
import importlib
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from tests import helpers

helpers.load_package()
regional = importlib.import_module(f"{helpers.PACKAGE}.engine.regional")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--features", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--share", type=float, default=.6)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if min(args.tokens, args.features, args.rank, args.iterations) <= 0 or not 0 < args.share < 1:
        parser.error("positive sizes/iterations and 0 < share < 1 are required")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(123)
    if args.device == "cpu":
        torch.set_num_threads(1)
    x = torch.randn(1, args.tokens, args.features, device=device, dtype=dtype)
    zeros = torch.zeros_like(x)
    first = torch.zeros(1, args.tokens, 1, device=device)
    first[:, :round(args.tokens * args.share)] = 1
    regional.CTX.token_masks = {"one": first, "two": 1 - first}
    regional.CTX.token_plans = {}
    adapters = {uid: ("lora", torch.randn(args.rank, args.features, device=device, dtype=dtype) * .01,
                      torch.randn(args.features, args.rank, device=device, dtype=dtype) * .01, .75)
                for uid in ("one", "two")}

    class Base(torch.nn.Module):
        def forward(self, x):
            return zeros
    layer = regional._patched_linear(Base())
    layer.regional_adapters = adapters

    def dense():
        y = zeros
        for uid, (_, a, b, scale) in adapters.items():
            delta = F.linear(F.linear(x, a), b)
            y = y + delta * (scale * regional.CTX.token_masks[uid].to(dtype))
        return y

    def sparse():
        return layer(x)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()
        elif args.device == "mps":
            torch.mps.synchronize()

    def measure(fn):
        for _ in range(5):
            fn()
        sync()
        durations = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            fn()
            sync()
            durations.append((time.perf_counter() - start) * 1000)
        return statistics.median(durations)

    with torch.inference_mode():
        torch.testing.assert_close(sparse(), dense(), rtol=.02, atol=.002)
        # Alternate measurements to reduce first/last-run bias.
        dense_first = measure(dense)
        sparse_first = measure(sparse)
        sparse_second = measure(sparse)
        dense_second = measure(dense)
        dense_ms = statistics.median([dense_first, dense_second])
        sparse_ms = statistics.median([sparse_first, sparse_second])
        sync()
    print(f"Synthetic two-LoRA overhead only: device={device}, dtype={dtype}, tokens={args.tokens}, features={args.features}, rank={args.rank}, share={args.share}")
    print(f"Dense: {dense_ms:.3f} ms; optimized: {sparse_ms:.3f} ms; speedup: {dense_ms / sparse_ms:.2f}x")
    print("Excludes the base model, attention, encoding, loading, and decoding. Does not predict total image generation time.")
    regional.CTX.clear()


if __name__ == "__main__":
    main()
