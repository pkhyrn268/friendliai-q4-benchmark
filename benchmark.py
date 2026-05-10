"""
vLLM vs Friendli Engine throughput benchmark.

Usage:
    python benchmark.py --demo                          # simulated data
    python benchmark.py --vllm-url ... --friendli-url ... --model <name>
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Fixed request shape removes workload variability from the benchmark.
CONCURRENCY_LEVELS    = [1, 2, 4, 8, 16, 32, 64]

INPUT_TOKENS          = 512
OUTPUT_TOKENS         = 256

REQUESTS_PER_LEVEL    = 100

# Warm-up requests are excluded to avoid startup-side effects
# such as lazy allocation or CUDA graph initialization.
WARMUP_REQUESTS       = 10

# Fixed-length synthetic prompt for reproducible benchmarking.
DUMMY_PROMPT = "word " * INPUT_TOKENS


@dataclass
class RequestResult:
    ttft: float           # TTFT(seconds)
    e2e_latency: float    # end-to-end latency(seconds)
    output_tokens: int    


@dataclass
class LevelResult:
    concurrency: int
    throughput: float     # tokens/sec  (primary metric)
    p95_ttft_ms: float    # milliseconds
    p50_ttft_ms: float
    raw: List[RequestResult] = field(default_factory=list)



async def send_request(session, endpoint: str, model: str) -> RequestResult:
    import aiohttp  # imported here so --demo can skip it

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": DUMMY_PROMPT}],
        "max_tokens": OUTPUT_TOKENS,
        "stream": True,
    }

    t0 = time.perf_counter()

    ttft = None
    output_tokens = 0

    async with session.post(
        f"{endpoint}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:

        resp.raise_for_status()

        # Streaming responses allow direct TTFT measurement.
        async for raw_line in resp.content:

            line = raw_line.decode("utf-8").strip()

            if not line.startswith("data:"):
                continue

            data_str = line[len("data:"):].strip()

            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            content = (
                (chunk.get("choices") or [{}])[0]
                .get("delta", {})
                .get("content", "")
            )

            if content:
                # First streamed token arrival time.
                if ttft is None:
                    ttft = time.perf_counter() - t0
                # Stream chunks are used as a lightweight proxy for generated tokens.
                output_tokens += 1

    e2e = time.perf_counter() - t0

    return RequestResult(
        ttft=ttft or e2e,
        e2e_latency=e2e,
        output_tokens=output_tokens,
    )


async def run_level(
    endpoint: str,
    model: str,
    concurrency: int,
    n_requests: int,
) -> List[RequestResult]:

    import aiohttp

    # Keep concurrency fixed so both engines are evaluated
    # under identical request pressure.
    sem = asyncio.Semaphore(concurrency)
    results: List[RequestResult] = []

    async def bounded(session):
        async with sem:
            return await send_request(session, endpoint, model)

    conn = aiohttp.TCPConnector(limit=concurrency * 2)

    async with aiohttp.ClientSession(connector=conn) as session:

        done = await asyncio.gather(
            *[bounded(session) for _ in range(n_requests)], 
            return_exceptions=True
        )

    for r in done:
        if isinstance(r, Exception):
            print(f"  [warn] {r}")
        else:
            results.append(r)
    return results



async def benchmark_engine(
    endpoint: str,
    model: str,
    label: str,
) -> List[LevelResult]:

    print(f"\n=== {label} @ {endpoint} ===")
    
    level_results = []

    for c in CONCURRENCY_LEVELS:
        
        print(
            f"  concurrency={c:3d} | warm-up...", 
            end="", 
            flush=True
        )

        await run_level(endpoint, model, c, WARMUP_REQUESTS)

        print(
            f" measuring ({REQUESTS_PER_LEVEL} req)...", 
            end="", 
            flush=True
        )

        t0 = time.perf_counter()
        results = await run_level(endpoint, model, c, REQUESTS_PER_LEVEL)
        elapsed = time.perf_counter() - t0

        if not results:
            print(" all failed, skipping.")
            continue

        total_out = sum(r.output_tokens for r in results)

        # Primary efficiency metric used for comparison.
        throughput = total_out / elapsed
        ttfts_ms = [r.ttft * 1000 for r in results]

        lr = LevelResult(
            concurrency=c,
            throughput=throughput,
            p95_ttft_ms=float(np.percentile(ttfts_ms, 95)),
            p50_ttft_ms=float(np.percentile(ttfts_ms, 50)),
            raw=results,
        )
        level_results.append(lr)

        print(
            f" throughput={throughput:7.1f} tok/s"
            f" | TTFT p95={lr.p95_ttft_ms:.0f}ms"
            )

    return level_results


def simulate_results(label: str) -> List[LevelResult]:

    random.seed(42 if "vLLM" in label else 7)

    # Synthetic scaling curves for offline demo mode.
    if "vLLM" in label:
        base = {1: 210, 2: 390, 4: 680, 8: 950, 16: 1120, 32: 1180, 64: 1140}
    else:  # Friendli Engine
        base = {1: 215, 2: 410, 4: 760, 8: 1180, 16: 1520, 32: 1830, 64: 2010}

    results = []

    for c in CONCURRENCY_LEVELS:
        noise = random.uniform(-0.02, 0.02)
        throughput = base[c] * (1 + noise)

        # Simulate per-request TTFT: grows with concurrency
        mean_ttft = (c ** 0.55) * (28 if "vLLM" in label else 22)
        ttfts = [
            max(
                5.0,
                random.gauss(mean_ttft, mean_ttft * 0.15),
            )
            for _ in range(REQUESTS_PER_LEVEL)
        ]

        results.append(LevelResult(
            concurrency=c,
            throughput=throughput,
            p95_ttft_ms=float(np.percentile(ttfts, 95)),
            p50_ttft_ms=float(np.percentile(ttfts, 50)),
        ))

        print(f"  [{label}] concurrency={c:3d} | throughput={throughput:7.1f} tok/s | TTFT p95={results[-1].p95_ttft_ms:.0f}ms")

    return results


def plot_results(
    vllm: List[LevelResult],
    friendli: List[LevelResult],
    output_path: str = "benchmark_result.png",
):
    fig, ax = plt.subplots(figsize=(10, 6))

    conc_v = [r.concurrency for r in vllm]
    tput_v = [r.throughput  for r in vllm]

    conc_f = [r.concurrency for r in friendli]
    tput_f = [r.throughput  for r in friendli]

    ax.plot(conc_v, tput_v, marker="o", linewidth=2.5, markersize=8,
            color="#4C72B0", label="vLLM")

    ax.plot(conc_f, tput_f, marker="s", linewidth=2.5, markersize=8,
            color="#DD8452", label="Friendli Engine")

    max_c = conc_f[-1]

    max_v = tput_v[-1]
    max_f = tput_f[-1]
    speedup = max_f / max_v

    # Highlight the scaling gap at high concurrency.
    ax.annotate(
        f"{speedup:.1f}× faster\nat concurrency {max_c}",
        xy=(max_c, max_f),
        xytext=(max_c * 0.55, max_f * 0.97),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        fontsize=11, color="black",
    )

    ax.set_xlabel("Concurrent Requests", fontsize=13)
    ax.set_ylabel("Throughput (tokens / sec)", fontsize=13)
    ax.set_title(
        "Inference Throughput vs. Concurrency\n"
        f"(input {INPUT_TOKENS} tokens · output {OUTPUT_TOKENS} tokens · {REQUESTS_PER_LEVEL} requests/level)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(CONCURRENCY_LEVELS)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nGraph saved → {output_path}")


def save_raw(
    vllm: List[LevelResult],
    friendli: List[LevelResult],
    path: str = "benchmark_raw.json",
):

    def _ser(results):
        return [
            {
                "concurrency": r.concurrency,
                "throughput_tokens_per_sec": round(r.throughput, 2),
                "p95_ttft_ms": round(r.p95_ttft_ms, 1),
                "p50_ttft_ms": round(r.p50_ttft_ms, 1),
            }
            for r in results
        ]
    with open(path, "w") as f:
        json.dump({"vllm": _ser(vllm), "friendli": _ser(friendli)}, f, indent=2)
    print(f"Raw numbers saved → {path}")


def parse_args():

    parser = argparse.ArgumentParser(
        description="vLLM vs Friendli Engine benchmark"
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
    )
    parser.add_argument(
        "--friendli-url",
        default="http://localhost:8001",
    )
    parser.add_argument(
        "--model",
        default="default",
    )
    parser.add_argument(
        "--output",
        default="benchmark_result.png",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate graph from simulated data",
    )
    return parser.parse_args()


async def main():

    args = parse_args()

    print("=== Benchmark Configuration ===")

    print(f"  input_tokens       = {INPUT_TOKENS}")
    print(f"  output_tokens      = {OUTPUT_TOKENS}")
    print(f"  requests_per_level = {REQUESTS_PER_LEVEL}")
    print(f"  warmup_requests    = {WARMUP_REQUESTS}")
    print(f"  concurrency_levels = {CONCURRENCY_LEVELS}")
    print(f"  mode               = {'DEMO (simulated)' if args.demo else 'REAL'}\n")

    if args.demo:
        print("Simulating vLLM results...")
        vllm_results = simulate_results("vLLM")

        print("\nSimulating Friendli Engine results...")
        friendli_results = simulate_results("Friendli Engine")
    else:
        vllm_results = await benchmark_engine(
            args.vllm_url,
            args.model,
            "vLLM",
        )

        friendli_results = await benchmark_engine(
            args.friendli_url,
            args.model,
            "Friendli Engine",
        )
        
    plot_results(vllm_results, friendli_results, args.output)
    save_raw(vllm_results, friendli_results)


if __name__ == "__main__":
    asyncio.run(main())
