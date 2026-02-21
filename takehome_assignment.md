# Serving Benchmark Guide — Qwen2.5-0.5B on Tesla T4

**Goal:** Maximize output token throughput (tok/s) for Qwen2.5-0.5B on a Tesla T4 GPU. The current vanilla baseline (on vllm serve bench given below) is **3,332 tok/s** — you have to better that significantly. 

## Prerequisites

- Python 3.10+, CUDA 12.4
- Colab GPU NVIDIA Tesla T4 (15 GB VRAM)
- Install vLLM:

```bash
pip install vllm
```

## Running the Serving Benchmark

```bash
vllm bench serve \
  --backend openai \
  --base-url http://localhost:8000/v1 \
  --endpoint /completions \
  --model Qwen/Qwen2.5-0.5B \
  --tokenizer Qwen/Qwen2.5-0.5B \
  --max-concurrency 50 \
  --num-prompts 200 \
  --ignore-eos \
  --random-input-len 512 \
  --random-output-len 512 \
  --save-result \
  --result-dir ./results
```

### Compare Against Baseline

| Run | Server Config | Output tok/s | TPOT | TTFT |
|-----|---------------|-------------|------|------|
| **Baseline (to beat)** | Default vLLM, concurrency=50 | **3,332** | 14.21 ms | 406 ms |
| Low concurrency | Default vLLM, concurrency=10 | 1,262 | 7.57 ms | 183 ms |
| Eager + spec decode | enforce-eager, ngram spec | 1,874 | 24.61 ms | 709 ms |

## Hints and Optimization Paths

### Understanding the Bottleneck

This workload is **memory-bandwidth bound**. The T4 has 320 GB/s bandwidth and reads ~0.98 GB of FP16 weights every decode step. At batch=50 the theoretical ceiling is ~16,327 tok/s — the baseline only reaches 20% of that. The gap might come from kernel launch overhead, attention computation, scheduling, and prefill interference.

The path to higher throughput:

1. **Reduce bytes read per step** — quantization (INT8 halves weight reads, INT4 quarters them)
2. **Reduce overhead per step** — fused kernels, better CUDA graphs
3. **Increase useful work per step** — speculative decoding (on real data, not random)
4. **Eliminate wasted work** — prefix caching, smarter scheduling

### What Works

- **Concurrency is the biggest lever.** Going from 10 to 50 concurrent requests gave 2.64x throughput. Higher batch sizes mean each weight read produces more tokens.
- **CUDA graphs + torch.compile are critical.** Disabling them (via `--enforce-eager`) dropped throughput by 44%. For a small 0.5B model, kernel launch overhead dominates.
- **Quantization is promising.** INT8 via bitsandbytes should theoretically double throughput since the T4 has 130 TOPS INT8 compute. Try: `--quantization bitsandbytes --load-format bitsandbytes`.
- **Tuning server config** with CUDA graphs still enabled (no `--enforce-eager`): try `--max-model-len 4096 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95`.

### What Doesn't Work out-of-box

- **Ngram speculative decoding on random/synthetic data** — near-zero acceptance rate, pure overhead. It requires real text with repeatable patterns.
- **Stacking optimizations that require `--enforce-eager`** — the CUDA graph penalty outweighs the gains on this small model.
- **FP8 KV cache** — T4 (compute capability 7.5) does not support FP8 natively. Dead end for this hardware.

### Key Tradeoff

More concurrency increases total throughput but hurts per-user latency. At concurrency=50, TPOT doubles vs concurrency=10. There is no free lunch on a single GPU.

## Scoring (these metrics will be produced via vllm bench serve command above)

| Metric | Weight | Constraint |
|--------|--------|------------|
| Output throughput (tok/s) | 40% | Higher is better |
| P99 TPOT | 20% | Must stay under 50 ms |
| P99 TTFT | 15% | Must stay under 2000 ms |
| Request success rate | 10% | Must be 100% |
| Code quality & documentation | 15% | Clean, reproducible, explained |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Server not ready | `run_bench.sh` waits up to 240s; check `curl http://localhost:8000/health` |
| OOM on GPU | Lower `--max-model-len` or `--gpu-memory-utilization` |
| Slow first run | Expected — model download and CUDA graph capture happen once |
| `protobuf` / `numpy` conflicts | `pip install "protobuf>=6.30.0" "numpy>=1.26.0,<2.2.0"` |
