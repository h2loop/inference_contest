# LLM Inference Optimization Challenge

## Qwen2.5-0.5B on Tesla T4 — Can You Beat 3,332 tok/s?

---

## 1. Problem Statement

You are given a single **NVIDIA Tesla T4 GPU** (15 GB VRAM) serving **Qwen2.5-0.5B** via vLLM or any serving engine. The target workload is **50 concurrent users**, each sending 512-token prompts and expecting 512-token completions.

Your goal: **maximize output token throughput (tok/s)** while keeping per-user latency acceptable.

The current best result achieved is **3,332 output tok/s**. Your challenge is to beat it.

---

## 2. Hardware Specifications

| Component | Spec |
|-----------|------|
| GPU | NVIDIA Tesla T4 |
| Compute Capability | 7.5 |
| VRAM | 15,360 MiB (GDDR6) |
| Memory Bandwidth | 320 GB/s |
| FP16 Compute | 65 TFLOPS |
| INT8 Compute | 130 TOPS |
| PCIe | Gen3 x16 (~15.75 GB/s) |
| CPU | Intel Xeon @ 2.30 GHz, 4 cores |
| System RAM | 14 GiB |
| CUDA Version | 12.4 |
| Driver | 550.90.07 |

## 3. Model Specifications

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) |
| Parameters | 0.49B (0.36B non-embedding) |
| Architecture | Transformer with RoPE, SwiGLU, RMSNorm |
| Layers | 24 |
| Attention Heads | 14 Q, 2 KV (GQA) |
| Head Dimension | 64 |
| Hidden Size | 896 |
| Max Context | 32,768 tokens |
| FP16 Weight Size | 0.98 GB |
| KV Cache per Token | 12,288 bytes (12 KB) |

## 4. Workload Definition

| Parameter | Value |
|-----------|-------|
| Concurrent Users | 50 |
| Input Tokens per Request | 512 |
| Output Tokens per Request | 512 |
| Total Tokens per Request | 1,024 |
| Dataset | Random (synthetic tokens) |
| Request Pattern | All requests arrive simultaneously (burst) |

---

## 5. Benchmark Results

All benchmarks used `vllm bench serve` with the OpenAI-compatible `/v1/completions` endpoint.

### Run A — Low Concurrency Baseline

**Server config:** Default vLLM (32K context, CUDA graphs ON, torch.compile ON)
**Benchmark:** 50 prompts, max concurrency = 10, 512 in / 512 out

| Metric | Value |
|--------|-------|
| Output Throughput | **1,262 tok/s** |
| Peak Output Throughput | 1,350 tok/s |
| Total Token Throughput | 2,525 tok/s |
| Request Throughput | 2.47 req/s |
| Duration | 20.3s |
| Mean TTFT | 183 ms |
| Median TTFT | 201 ms |
| P99 TTFT | 327 ms |
| Mean TPOT | 7.57 ms |
| P99 TPOT | 7.82 ms |
| Mean ITL | 7.58 ms |
| P99 ITL | 8.67 ms |

### Run B — Concurrency = 50 (Best Result)

**Server config:** Default vLLM (32K context, CUDA graphs ON, torch.compile ON)
**Benchmark:** 200 prompts, max concurrency = 50, 512 in / 512 out

| Metric | Value |
|--------|-------|
| Output Throughput | **3,332 tok/s** |
| Peak Output Throughput | 4,100 tok/s |
| Total Token Throughput | 6,664 tok/s |
| Request Throughput | 6.51 req/s |
| Duration | 30.7s |
| Mean TTFT | 406 ms |
| Median TTFT | 350 ms |
| P99 TTFT | 1,180 ms |
| Mean TPOT | 14.21 ms |
| P99 TPOT | 14.80 ms |
| Mean ITL | 14.21 ms |
| P99 ITL | 84.71 ms |

### Run C — All "Optimizations" Stacked (Regressed)

**Server config:** max-model-len=4096, max-num-batched-tokens=4096, gpu-memory-utilization=0.95, speculative decoding (ngram, 5 tokens), **enforce-eager (CUDA graphs OFF, torch.compile OFF)**
**Benchmark:** 200 prompts, max concurrency = 50, 512 in / 512 out

| Metric | Value |
|--------|-------|
| Output Throughput | **1,874 tok/s** |
| Peak Output Throughput | 2,200 tok/s |
| Total Token Throughput | 3,748 tok/s |
| Request Throughput | 3.66 req/s |
| Duration | 54.6s |
| Mean TTFT | 709 ms |
| Median TTFT | 178 ms |
| P99 TTFT | 2,730 ms |
| Mean TPOT | 24.61 ms |
| P99 TPOT | 26.22 ms |
| Mean ITL | 25.36 ms |
| P99 ITL | 49.80 ms |

### Run D — Unlimited Concurrency (Burst, shorter output)

**Server config:** Default vLLM (32K context, CUDA graphs ON, torch.compile ON)
**Benchmark:** 100 prompts, no concurrency limit, 512 in / **128 out**

| Metric | Value |
|--------|-------|
| Output Throughput | **2,997 tok/s** |
| Peak Output Throughput | 6,199 tok/s |
| Total Token Throughput | 14,986 tok/s |
| Request Throughput | 23.42 req/s |
| Duration | 4.3s |
| Mean TTFT | 1,305 ms |
| P99 TTFT | 2,341 ms |
| Mean TPOT | 22.35 ms |
| P99 TPOT | 29.49 ms |

---

## 6. Comparison Summary

| Run | Config | Concurrency | Output tok/s | TPOT (ms) | TTFT (ms) | Speedup vs A |
|-----|--------|-------------|-------------|-----------|-----------|-------------|
| A | Default server | 10 | 1,262 | 7.57 | 183 | 1.0x |
| **B** | **Default server** | **50** | **3,332** | **14.21** | **406** | **2.64x** |
| C | Eager + spec decode | 50 | 1,874 | 24.61 | 709 | 1.48x |
| D | Default server | unlimited | 2,997 | 22.35 | 1,305 | 2.38x |

---

## 7. Theoretical Analysis

### Why This Workload is Memory-Bandwidth Bound

Every autoregressive decode step reads the entire model weights from VRAM to produce one token per sequence. The key metric is **arithmetic intensity** — how many FLOPs you perform per byte read from memory.

```
Arithmetic Intensity = batch_size * 2 * params / (2 * params)  [for FP16]
                     = batch_size  (FLOPs/byte)

T4 Compute/Bandwidth Ratio = 65 TFLOPS / 320 GB/s = 203 FLOPs/byte
```

| Batch Size | Arithmetic Intensity | Regime | Theoretical Max tok/s |
|------------|---------------------|--------|----------------------|
| 1 | 1 | Memory-bound | 327 |
| 10 | 10 | Memory-bound | 3,265 |
| 50 | 50 | Memory-bound | 16,327 |
| 203 | 203 | Crossover point | 66,327 |
| 500 | 500 | Compute-bound | 66,327 |

At batch=50, the theoretical maximum is **16,327 tok/s**. We achieved **3,332 tok/s** — roughly **20.4% of theoretical**.

### Where the 79.6% Gap Goes

1. **Attention computation** — scales with sequence length, not just model size. At 512+ tokens, FlashInfer attention is a significant fraction of each step.
2. **Kernel launch overhead** — ~150 kernel launches per decode step, each with ~5-10 us overhead.
3. **Chunked prefill interference** — prefill chunks (max 2048 tokens) interleave with decode steps, stealing GPU cycles.
4. **Scheduling and HTTP overhead** — Python scheduler, tokenizer, and API server add latency between GPU steps.
5. **CUDA graph padding** — batch sizes that don't match captured graph sizes (1, 2, 4, 8, 16, 24, 32, 40, 48, 56...) waste compute on padded slots.

### KV Cache Budget

```
KV cache per token = 2 * 24 layers * 2 KV heads * 64 dim * 2 bytes = 12,288 bytes
50 users * 1024 tokens = 0.63 GB KV cache needed
Available VRAM for KV (FP16): ~13.9 GB  (after model + overhead)
Max tokens in KV cache: ~1,130,000
Max concurrent 1024-token sequences: ~1,103
```

KV cache is **not** the bottleneck for this workload. There is ample VRAM.

---

## 8. What We Learned

### Finding 1: Concurrency is the single biggest lever (2.64x)

Going from 10 to 50 concurrent requests increased throughput from 1,262 to 3,332 tok/s. Each weight read from VRAM now produces 5x more tokens. The cost: per-token latency doubled (7.57ms to 14.21ms TPOT).

### Finding 2: CUDA graphs + torch.compile are critical (1.78x impact)

Disabling them (via enforce-eager, required by speculative decoding on this vLLM version) **dropped throughput by 44%** at the same concurrency. For a small 0.5B model, kernel launch overhead dominates — CUDA graphs eliminate it by replaying captured GPU command sequences.

### Finding 3: Speculative decoding (ngram) doesn't help on random data

Ngram prompt lookup guesses next tokens from patterns in the input. Random synthetic tokens have no patterns. Acceptance rate was near zero, adding overhead for no benefit. This would behave differently on real-world text.

### Finding 4: The throughput-latency tradeoff is inescapable on one GPU

| Concurrency | Output tok/s | Per-user tok/s | TPOT |
|-------------|-------------|---------------|------|
| 10 | 1,262 | 126 | 7.57 ms |
| 50 | 3,332 | 67 | 14.21 ms |

Total throughput goes up, but each user's experience gets slower. There is no free lunch.

---

## 9. The Challenge: Optimization Tasks

Below are concrete optimization paths, ordered by expected difficulty. Each one should be benchmarked using the same workload (50 concurrent users, 512 in / 512 out, 200 prompts) and compared against **Run B (3,332 tok/s)** as the baseline to beat.

### Level 1: Server Configuration (Easy)

**Task 1.1 — Optimal max-model-len + max-num-batched-tokens WITHOUT eager mode**

The combination of `--max-model-len 4096 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95` was never tested with CUDA graphs enabled (no speculative decoding). This should give the memory savings without the eager-mode penalty.

```bash
vllm serve Qwen/Qwen2.5-0.5B \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95
```

**Expected gain:** 5-15% over Run B (more efficient batching, less KV waste).

**Task 1.2 — Tune max-num-seqs**

Default is unset. Explicitly setting `--max-num-seqs 256` or higher may allow the scheduler to batch more aggressively.

**Task 1.3 — Block size tuning**

Test `--block-size 16` vs default. Smaller blocks reduce KV cache fragmentation but increase management overhead.

### Level 2: Quantization (Medium)

**Task 2.1 — INT8 weight quantization via bitsandbytes**

Halves the model weight size from 0.98 GB to 0.49 GB. Since the workload is memory-bandwidth bound, this theoretically doubles the decode throughput ceiling.

```bash
pip install bitsandbytes
vllm serve Qwen/Qwen2.5-0.5B \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95
```

**Expected gain:** 1.5-2x throughput (memory bandwidth doubling).

**Caveat:** T4 has native INT8 tensor cores (130 TOPS), but verify bitsandbytes actually uses them vs CPU dequantization.

**Task 2.2 — GPTQ/AWQ INT4 quantization**

Use a pre-quantized model like `OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc`. 4x weight compression, but quality may degrade on a model this small.

**Task 2.3 — FP8 KV cache**

The T4 (compute capability 7.5) does **not** support FP8 natively (needs CC 8.0+). This is a dead end for this hardware. Document why.

### Level 3: Speculative Decoding (Medium-Hard)

**Task 3.1 — Ngram speculative decoding on real-world data**

Re-run the ngram spec decode test but with ShareGPT or a real dataset where input/output overlap exists:

```bash
vllm serve Qwen/Qwen2.5-0.5B \
  --max-model-len 4096 \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'
```

Benchmark with `--dataset-name sharegpt --dataset-path <path>`.

**Key question:** Can speculative decoding work WITHOUT enforce-eager on newer vLLM versions or with different attention backends?

**Task 3.2 — Draft model speculative decoding**

Use a tiny draft model to propose tokens. This doesn't require enforce-eager:

```bash
vllm serve Qwen/Qwen2.5-0.5B \
  --speculative-config '{"model": "Qwen/Qwen2.5-0.5B", "num_speculative_tokens": 3}'
```

(Using the same model as draft is a baseline; ideally find/train a smaller draft model.)

### Level 4: Kernel Engineering (Hard)

**Task 4.1 — Custom Triton fused kernels**

Write a fused Triton kernel that combines RMSNorm + QKV projection into a single kernel, eliminating one intermediate VRAM read/write per layer. With 24 layers, this removes 24 kernel launches and 24 intermediate tensor materializations per decode step.

**Target file:** Write a custom Triton kernel and register it as a vLLM custom op.

**Expected gain:** 10-20% throughput improvement.

**Task 4.2 — Persistent CUDA kernel for decode**

Instead of launching 150+ kernels per decode step, write a single persistent kernel that stays resident on SMs and processes the entire decode step. This eliminates all kernel launch overhead.

**Reference:** [FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html), [TensorRT-LLM fused kernels](https://github.com/NVIDIA/TensorRT-LLM).

### Level 5: Systems Engineering (Hard)

**Task 5.1 — Prefix caching with shared system prompts**

Design a workload where all 50 users share a common 256-token system prompt. Measure the throughput gain from prefix caching (already enabled by default). The prefill for the shared prefix should happen once, not 50 times.

**Task 5.2 — Data parallelism across request streams**

Investigate if splitting the 50 concurrent users into multiple vLLM engine instances (e.g., 2x25) with different scheduling strategies improves aggregate throughput.

**Task 5.3 — CPU offloading for KV cache**

With `--swap-space 4`, offload idle KV cache to CPU RAM. This allows serving more concurrent long-context requests than VRAM alone supports. Measure the throughput impact.

### Level 6: Alternative Backends (Expert)

**Task 6.1 — TensorRT-LLM comparison**

Build and benchmark the same model with NVIDIA TensorRT-LLM, which uses fused kernels and more aggressive optimization. Compare throughput numbers.

**Task 6.2 — SGLang comparison**

Benchmark with [SGLang](https://github.com/sgl-project/sglang), which has different scheduling and RadixAttention for prefix caching. Compare throughput.

**Task 6.3 — llama.cpp GGUF comparison**

Convert to GGUF format and benchmark with llama.cpp server. Compare throughput and latency, especially for INT4 quantized inference.

---

## 10. Scoring Criteria

| Metric | Weight | How to Measure |
|--------|--------|---------------|
| Output Throughput (tok/s) | 40% | `vllm bench serve` output_throughput |
| P99 TPOT (ms) | 20% | Must stay under 50ms |
| P99 TTFT (ms) | 15% | Must stay under 2000ms |
| Request Success Rate | 10% | Must be 100% |
| Code Quality & Documentation | 15% | Clean, reproducible, explained |

### Benchmark Command (Use This Exactly)

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

---

## 11. Reproducing the Baseline

### Environment Setup

```bash
# Python 3.10, CUDA 12.4
pip install vllm
```

### Start Baseline Server (Run B — 3,332 tok/s)

```bash
vllm serve Qwen/Qwen2.5-0.5B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
```

### Start "Optimized" Server (Run C — 1,874 tok/s, regression)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vllm serve Qwen/Qwen2.5-0.5B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'
```

---

## 12. Raw Benchmark Data

All JSON result files are in the `results/` directory:

- `results/step1_baseline_concurrency50.json` — Run B (best: 3,332 tok/s)
- `results/step1to5_all_optimized.json` — Run C (regressed: 1,874 tok/s)

Earlier runs in the project root:
- `openai-infqps-concurrency10-Qwen2.5-0.5B-20260209-113242.json` — Run A (1,262 tok/s)
- `vllm-infqps-Qwen2.5-0.5B-20260209-112341.json` — Run D (2,997 tok/s)

---

## 13. Key Insight for Challengers

The biggest lesson from this exercise: **understanding the hardware bottleneck matters more than stacking optimization flags.**

This workload is memory-bandwidth bound. The T4 reads 0.98 GB of weights per decode step and has 320 GB/s bandwidth. At batch=50, the theoretical ceiling is 16,327 tok/s. We're at 3,332 — **20% efficiency**. The remaining 80% is overhead from kernel launches, attention computation, scheduling, and prefill interference.

The path to higher throughput is:
1. **Reduce bytes read per step** (quantization: INT8 halves it, INT4 quarters it)
2. **Reduce overhead per step** (fused kernels, persistent kernels, better CUDA graphs)
3. **Increase useful work per step** (speculative decoding on real data)
4. **Eliminate wasted work** (prefix caching, smarter scheduling)

Good luck.

---

*Generated: 2026-02-09 | vLLM v0.15.1 | Tesla T4 | Qwen2.5-0.5B*
