"""
Performance metrics for evaluating model inference latency and VRAM usage.
@author: Stephen Krol 27/01/2026
"""

import torch
import time
from transformers import PreTrainedModel

def timed_inference(model: PreTrainedModel, inputs: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Docstring for timed_inference function, which measures the time taken for model inference.
    Import that cuda is synchronized before and after inference to get accurate timing due
    to asynchronous nature of GPU operations.
    @author: sjkro1

    :param model: transformer model to be evaluated
    :type model: PreTrainedModel
    :param inputs: input tensor for the model
    :type inputs: torch.Tensor

    :return: tuple of (model outputs, time taken in seconds)
    :rtype: tuple[torch.Tensor, float]
    """
    torch.cuda.synchronize()
    start = time.perf_counter()

    outputs = model.generate(**inputs)

    torch.cuda.synchronize()
    end = time.perf_counter()

    return outputs, end - start

def benchmark(model: PreTrainedModel, inputs: torch.Tensor, runs: int = 10, warmup: int = 3) -> dict[str, float]:
    """
    Docstring for benchmark function that runs multiple inferences on a model and computes
    performance metrics such as mean, p50, and p95 inference times. It includes a warm-up phase
    to stabilize performance before measurements.
    @author: sjkro1

    :param model: transformer model to be evaluated
    :type model: PreTrainedModel
    :param inputs: input tensor for the model
    :type inputs: torch.Tensor
    :param runs: number of inference runs to perform
    :type runs: int
    :param warmup: number of warm-up runs to perform
    :type warmup: int

    :return: dictionary with mean, p50, and p95 inference times in seconds
    :rtype: dict[str, float]
    """
    times = []
    peak_allocated = []

    # Warm-up
    for _ in range(warmup):
        _ = model.generate(**inputs)
    torch.cuda.synchronize()

    for _ in range(runs):
        torch.cuda.reset_peak_memory_stats()


        _, t = timed_inference(model, inputs)
        times.append(t)

    return {
        "mean_s": sum(times) / len(times),
        "p50_s": sorted(times)[len(times)//2],
        "p95_s": sorted(times)[int(len(times)*0.95)],
    }

def benchmark_latency_and_vram(
    model: PreTrainedModel,
    inputs: torch.Tensor,
    runs: int = 10,
    warmup: int = 3,
    ) -> dict[str, float]:

    """
    Docstring for benchmark_latency_and_vram. This function benchmarks a model's 
    inference latency and VRAM usage over multiple runs, including a warm-up phase.
    It returns latency percentiles (p50, p95, mean) and VRAM  allocation statistics
    (p50, p95, peak).
    @author: sjkro1
    
    :param model: Model to benchmark
    :type model: PreTrainedModel
    :param inputs: Input tensor for the model
    :type inputs: torch.Tensor
    :param runs: Number of inference runs to perform
    :type runs: int
    :param warmup: Number of warm-up runs to perform
    :type warmup: int

    :return: Dictionary with latency and VRAM usage metrics
    :rtype: dict[str, float]
    """

    times = []
    peak_allocated = []

    # Warm-up (not measured)
    for _ in range(warmup):
        _ = model.generate(**inputs)
    torch.cuda.synchronize()

    for _ in range(runs):
        # Reset per-run memory peak
        torch.cuda.reset_peak_memory_stats()

        # CUDA-safe timing
        torch.cuda.synchronize()
        start = time.perf_counter()

        _ = model.generate(**inputs)

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)
        peak_allocated.append(
            torch.cuda.max_memory_allocated() / 1024**2
        )

    # Sort for percentile calculation
    times.sort()
    peak_allocated.sort()

    return {
        # Latency
        "latency_p50_s": times[len(times) // 2],
        "latency_p95_s": times[int(len(times) * 0.95)],
        "latency_mean_s": sum(times) / len(times),

        # VRAM (true memory cost)
        "vram_alloc_p50_MB": peak_allocated[len(peak_allocated) // 2],
        "vram_alloc_p95_MB": peak_allocated[int(len(peak_allocated) * 0.95)],
        "vram_alloc_peak_MB": max(peak_allocated)
    }

if __name__ == "__main__":
    # Example usage (requires a model and inputs)
    print(benchmark_latency_and_vram)
