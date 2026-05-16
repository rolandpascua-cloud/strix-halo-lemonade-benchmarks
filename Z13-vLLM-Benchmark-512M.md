# Z13 Local AI Stack — vLLM Benchmark Report (Baseline: 512 MB UMA Buffer)

> **Part 3 of 4** in the *Z13 Local AI Stack Benchmark Series* — evaluating vLLM serving on AMD Strix Halo under the same 128 GB unified-memory baseline used for the Lemonade reports.
>
> **Series:** [Lemonade · 512M UMA](Z13-Lemonade-Benchmark-512M.md) · [Lemonade · 96 GB VRAM](Z13-Lemonade-Benchmark-96G.md) · [vLLM · 512M UMA *(this report)*](Z13-vLLM-Benchmark-512M.md) · [vLLM · 96 GB VRAM](Z13-vLLM-Benchmark-96G.md)

| Field | Value |
|---|---|
| **Author** | Roland Pascua |
| **Platform** | ASUS ROG Flow Z13 GZ302EA · AMD Ryzen AI MAX+ 395 (Strix Halo) |
| **GPU** | Radeon 8060S (gfx1151 / RDNA 3.5 · 40 CUs · 2900 MHz boost) |
| **Memory** | 128 GB LPDDR5x-8533 unified · ~273 GB/s bandwidth |
| **iGPU VRAM (BIOS)** | UMA Buffer = **512 MB** (dynamic GTT allocation up to ~105 GiB) |
| **OS / Kernel** | Fedora 43 · Linux 7.0.7-100.fc43.x86_64 |
| **vLLM Version** | v0.19.2rc1.dev113+g6aa057c9d (TheRock nightly, `kyuz0/vllm-therock-gfx1151:stable`) |
| **ROCm** | 7.12.0 (bundled in container) |
| **Attention Backend** | `TRITON_ATTN` (Flash Attention not available on gfx1151) |
| **Date** | 2026-05-16 |
| **Models Tested** | 8 successful + 1 failed (Mixtral-8x7B OOM) |
| **License** | Report under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · scripts under MIT |

---

## 1. Executive Summary

Eight models were successfully benchmarked under vLLM serving, plus one failure (Mixtral-8x7B OOM). All measurements are **single-request** throughput — vLLM's architecture is optimized for concurrent batch workloads, so these numbers represent its floor, not its ceiling.

Three findings dominate:

1. **vLLM single-request throughput is 51–85% lower than GGUF/llama.cpp across all models.** This is expected and not a vLLM defect — llama.cpp is a single-stream engine while vLLM is a production server optimized for multi-request batching. The comparison is apples-to-oranges for production use.

2. **LFM2 SSM models pay a disproportionate vLLM penalty.** LFM2-8B-A1B drops 85% vs its GGUF baseline — the `--enforce-eager` requirement (SSM architecture, no HIP graph capture) eliminates vLLM's graph optimization entirely. In GGUF, LFM2-8B-A1B is the #3 fastest model; in vLLM it ranks #6.

3. **gpt-oss-120b (120B MXFP4 MoE) successfully ran under vLLM** — 7.5 tok/s single-request from a 70.29 GiB model loaded entirely in unified memory. Under batch load, expect significantly higher effective throughput.

**Methodology:** 3 measured runs + 1 warmup · median reported · client-side timing · `max_tokens=256` · greedy decoding

**Prompt:** "Explain the difference between machine learning and deep learning in exactly one sentence."

---

## 2. Results — Ranked by Generation Throughput

| Rank | Model | Gen tok/s | Prompt tok/s | TTFT | GGUF baseline | Δ vs GGUF |
|---:|---|---:|---:|---:|---:|---|
| 1 | **LFM2-1.2B** | 76.6 | 572.0 | 0.042s | 183.5 | −106.9 (−58%) |
| 2 | **Qwen3.5-0.8B** | 71.2 | 706.0 | 0.044s | 191.6 | −120.4 (−63%) |
| 3 | **Qwen3.5-2B** | 44.6 | 377.5 | 0.082s | 101.3 | −56.7 (−56%) |
| 4 | **Ministral-3-3B-BF16** | 31.1 | 270.1 | 0.067s | 63.8 | −32.7 (−51%) |
| 5 | **granite-4.0-h-tiny** | 30.6 | 282.7 | 0.159s | 104.8 | −74.2 (−71%) |
| 6 | **LFM2-8B-A1B** | 23.0 | 145.2 | 0.165s | 153.6 | −130.6 (−85%) |
| 7 | **gpt-oss-20b** | 11.1 | 440.1 | 0.184s | 66.1 | −55.0 (−83%) |
| 8 | **gpt-oss-120b** | 7.5 | 289.7 | 0.280s | 47.9 | −40.4 (−84%) |

### Failed Loads

| Model | Reason |
|---|---|
| Mixtral-8x7B-Instruct-v0.1 | Engine core init failed (rc=1) — ~93 GB BF16 weights leave insufficient KV cache headroom at `gpu_memory_utilization=0.99` |

---

## 3. vLLM vs GGUF Analysis

### Why vLLM Is Slower on Single Requests

llama.cpp (GGUF backend) is a single-stream inference engine: it dedicates all GPU bandwidth to one request at a time. vLLM is a continuous-batching server built to overlap many concurrent requests using PagedAttention and iteration-level scheduling. The single-request measurement below captures vLLM's minimum case, not its intended operating mode.

| Metric | GGUF / llama.cpp | vLLM |
|---|---|---|
| Serving model | Single-stream | Continuous batching (PagedAttention) |
| Optimized for | Minimum latency, 1 user | Maximum throughput, N concurrent users |
| Overhead on single request | Near zero | Scheduling + KV page management |
| Graph capture (CUDA/HIP) | Not applicable | Yes — improves throughput under load |
| Quantization support | Q4_K_M, Q4_K_XL, MXFP4, GGUF native | BF16, FP16, MXFP4 safetensors |

### Per-Model Comparison

| Model | GGUF tok/s | vLLM tok/s | Δ | vLLM notes |
|---|---:|---:|---|---|
| LFM2-1.2B | 183.5 | 76.6 | −58% | `--enforce-eager` (SSM) |
| Qwen3.5-0.8B | 191.6 | 71.2 | −63% | Standard Transformer, graph capture |
| Qwen3.5-2B | 101.3 | 44.6 | −56% | Standard Transformer, graph capture |
| Ministral-3-3B-BF16 | 63.8 | 31.1 | −51% | **Smallest vLLM penalty** — BF16 weights |
| granite-4.0-h-tiny | 104.8 | 30.6 | −71% | Hybrid attention; torch.compile |
| LFM2-8B-A1B | 153.6 | 23.0 | −85% | **Largest penalty** — `--enforce-eager` (SSM) |
| gpt-oss-20b | 66.1 | 11.1 | −83% | MXFP4 MoE; CUDA graph capture (83 graphs) |
| gpt-oss-120b | 47.9 | 7.5 | −84% | MXFP4 MoE; 70.29 GiB loaded; max_len=8192 |

### When to Use vLLM vs GGUF on This Platform

| Use Case | Recommended Backend | Reason |
|---|---|---|
| Single user, minimum latency | GGUF (llama.cpp) | Lower overhead, higher single-stream tok/s |
| Multiple concurrent users | vLLM | Continuous batching recovers throughput loss |
| API server for agents/apps | vLLM | OpenAI-compatible, production-grade scheduler |
| Local chat / IDE copilot | GGUF | No server overhead; simpler deployment |
| Large MoE models (MoE batching) | vLLM | PagedAttention handles expert routing efficiently at scale |
| Streaming with structured output | vLLM | Native guided decoding, streaming API |

---

## 4. Model Profiles

### LFM2-1.2B
- **Architecture:** SSM MoE (Liquid AI, non-Transformer) — `Lfm2ForCausalLM`
- **vLLM gen tok/s:** 76.6 · **GGUF:** 183.5 (−58%)
- **vLLM flags:** `--enforce-eager` (HIP graph capture incompatible with SSM recurrence)
- **TTFT:** 0.042s — fastest in the vLLM benchmark
- **Notes:** Despite the SSM penalty, still the fastest model under vLLM. Ranks #2 in the GGUF benchmark at 183.5 tok/s. Tiny 0.73 GB footprint; 1.2B total parameters, all active.

### Qwen3.5-0.8B
- **Architecture:** Dense Transformer — `Qwen3_5ForConditionalGeneration`
- **vLLM gen tok/s:** 71.2 · **GGUF:** 191.6 (−63%)
- **vLLM flags:** None (standard graph capture)
- **TTFT:** 0.044s · **Prompt tok/s:** 706.0 — fastest prompt throughput in vLLM benchmark
- **Notes:** GGUF rank #1 at 191.6 tok/s; vLLM rank #2. At 0.8B / ~0.5 GB, suitable for classification, routing, and high-volume batch use. Qwen3.5 architecture improvements over Qwen3 transfer to vLLM.

### Qwen3.5-2B
- **Architecture:** Dense Transformer — `Qwen3_5ForConditionalGeneration`
- **vLLM gen tok/s:** 44.6 · **GGUF:** 101.3 (−56%)
- **vLLM flags:** None (standard graph capture)
- **TTFT:** 0.082s · **Prompt tok/s:** 377.5
- **Notes:** Uses a Gated Linear Attention (GDN) hybrid attention layer — vLLM's startup log shows "Triton/FLA GDN prefill kernel". At 1.34 GB, practical for general assistant workloads.

### Ministral-3-3B-BF16
- **Architecture:** Dense Transformer (Mistral) — `MistralForCausalLM`
- **vLLM gen tok/s:** 31.1 · **GGUF:** 63.8 (−51%)
- **vLLM flags:** None
- **TTFT:** 0.067s · **Prompt tok/s:** 270.1
- **Notes:** The −51% penalty is the **smallest vLLM overhead of all models tested** — BF16 full-precision weights provide a clean mapping to vLLM's compute path with no quantization conversion overhead. GGUF baseline uses Q4_K_M, so the BF16 vLLM model is higher quality than its GGUF counterpart while still achieving 31 tok/s. Model loaded 7.28 GiB into GTT.

### granite-4.0-h-tiny
- **Architecture:** Hybrid attention (IBM Granite) — `GraniteMoeHybridForCausalLM`
- **vLLM gen tok/s:** 30.6 · **GGUF:** 104.8 (−71%)
- **vLLM flags:** None (`torch.compile`; no enforce-eager needed)
- **TTFT:** 0.159s · **Prompt tok/s:** 282.7
- **Notes:** GGUF rank #4 at 104.8 tok/s — a standout performer in the GGUF benchmark. The higher vLLM penalty (−71%) likely stems from the hybrid attention architecture requiring more compilation steps. Still functional, ranks #5 by vLLM gen throughput.

### LFM2-8B-A1B
- **Architecture:** SSM MoE (Liquid AI) — `Lfm2MoeForCausalLM`; 8B total / ~1B active
- **vLLM gen tok/s:** 23.0 · **GGUF:** 153.6 (−85%)
- **vLLM flags:** `--enforce-eager` (SSM architecture, no graph capture)
- **TTFT:** 0.165s · **Prompt tok/s:** 145.2
- **Notes:** The **largest vLLM penalty of any model** (−85%). Under GGUF this is the #3 fastest model at 153.6 tok/s; under vLLM it falls to #6 at 23.0 tok/s. `--enforce-eager` forces eager execution, bypassing all HIP graph optimization. The GGUF backend handles SSM recurrence natively without this constraint.

### gpt-oss-20b
- **Architecture:** Dense MXFP4 MoE — `GptOssForCausalLM`; 20B parameters
- **vLLM gen tok/s:** 11.1 · **GGUF:** 66.1 (−83%)
- **vLLM flags:** `--trust-remote-code`
- **TTFT:** 0.184s · **Prompt tok/s:** 440.1
- **Notes:** Custom OpenAI architecture requiring trust-remote-code. vLLM performed CUDA graph capture (83 graphs) at startup — primary driver of the long startup time (~5 min). MXFP4 weights (~14.7 GiB loaded); uses "TRITON Mxfp4 MoE backend". Despite the single-request slowdown, prompt throughput (440.1 tok/s) is the second highest in the vLLM benchmark.

### gpt-oss-120b
- **Architecture:** Dense MXFP4 MoE — `GptOssForCausalLM`; 120B parameters
- **vLLM gen tok/s:** 7.5 · **GGUF:** 47.9 (−84%)
- **vLLM flags:** `--trust-remote-code` · `--max-model-len 8192` (reduced from 32768 to fit KV cache)
- **TTFT:** 0.280s · **Prompt tok/s:** 289.7
- **Notes:** Largest model successfully benchmarked. Downloaded ~61 GB during the run (model was not pre-cached); 15 safetensors shards loaded in 37.77s. GPU footprint: **70.29 GiB in GTT** — nearly two-thirds of the 104.9 GiB unified pool. KV cache limited to 8192 tokens by memory budget. Under concurrent load, this model's MXFP4 MoE backend is expected to scale efficiently.

### Mixtral-8x7B-Instruct-v0.1 (FAILED)
- **Architecture:** Dense BF16 MoE — 46.7B total / ~13B active per token; 8 experts, top-2 routing
- **vLLM flags:** `--enforce-eager` · `--gpu-memory-utilization 0.99` · `--max-model-len 4096` · `--max-num-seqs 4`
- **Failure:** Engine core init failed (rc=1). BF16 weights require ~93 GB; at 0.99 × 104.9 GiB = 103.8 GiB total budget, only ~11 GB would remain for KV cache and activations — insufficient for initialization.
- **GGUF result:** Mixtral-8x7B was not included in the GGUF benchmark.

---

## 5. Hardware Context

| Component | Specification |
|---|---|
| **Platform** | ASUS ROG Flow Z13 GZ302EA (2-in-1 tablet/laptop) |
| **CPU** | AMD Ryzen AI MAX+ 395 (Strix Halo) |
| **CPU cores** | 16 cores / 32 threads (Zen 5 architecture) |
| **CPU frequency** | Base ~2.0 GHz · Boost up to 5187 MHz |
| **System RAM** | 128 GB LPDDR5x unified memory (shared pool: CPU + GPU + NPU) |
| **System tuning** | `tuned-adm` active profile: `accelerator-performance` |
| **BIOS UMA Buffer** | **512 MB** (dynamic GTT allocation up to ~105 GiB) |
| **vLLM backend** | `TRITON_ATTN` (required; Flash Attention not available on gfx1151) |

`rocm-smi` reported memory state during this run:

```
VRAM Total Memory (B):      536,870,912    (~512 MiB — UMA carve-out)
VIS_VRAM Total Memory (B):  536,870,912
GTT  Total Memory (B):  113,246,208,000    (~105 GiB — dynamically allocatable from unified pool)
```

### GPU (Integrated, Radeon 8060S)

| Attribute | Value |
|---|---|
| **GPU model** | AMD Radeon 8060S |
| **GPU codename** | gfx1151 (RDNA 3.5 architecture) |
| **Compute Units** | 40 CUs |
| **Boost clock** | 2900 MHz |
| **Memory bandwidth** | ~273 GB/s (LPDDR5x-8533 ×256-bit) |

---

## 6. Software & Stack

### vLLM Environment

| Component | Version / Detail |
|---|---|
| **vLLM** | 0.19.2rc1.dev113+g6aa057c9d.d20260422 |
| **Container image** | `docker.io/kyuz0/vllm-therock-gfx1151:stable` |
| **Container runtime** | Podman (Toolbx) on Fedora 43 |
| **ROCm** | 7.12.0 (TheRock nightly build, bundled in container) |
| **Attention backend** | `--attention-backend TRITON_ATTN --mm-encoder-attn-backend TRITON_ATTN` (mandatory for gfx1151; Flash Attention not supported) |
| **Compile cache** | `VLLM_DISABLE_COMPILE_CACHE=1` (set for all runs to prevent stale graph reuse) |
| **Supported architectures** | `Lfm2ForCausalLM`, `Lfm2MoeForCausalLM`, `GraniteMoeHybridForCausalLM`, `GptOssForCausalLM`, `Qwen3_5ForConditionalGeneration` |

### Host OS

| Component | Version |
|---|---|
| **OS** | Fedora 43 (KDE Plasma desktop) |
| **Kernel** | 7.0.7-100.fc43.x86_64 |
| **Host ROCm** | 6.4.2 (system-level; container uses ROCm 7.12.0) |

### GGUF Baseline Stack (for comparison)

| Component | Version |
|---|---|
| **Lemonade Server** | v10.4.0 (snap rev 177) |
| **GPU backend** | llamacpp `rocm-preview` + Vulkan |
| **API endpoint** | `http://localhost:13305/api/v1` |

---

## 7. Benchmark Methodology

### Setup

Each model was launched via `vllm serve` inside the vllm Toolbx container, one at a time. The benchmark suite waited for the `/v1/models` health endpoint to return HTTP 200 before proceeding. After benchmarking, the vLLM server process group was killed and the suite waited for GTT memory to drop below 80 GiB before starting the next model.

### Measurement

| Parameter | Value |
|---|---|
| Warmup runs | 1 (discarded) |
| Measured runs | 3 |
| Reported value | Median of 3 runs |
| `max_tokens` | 256 |
| `temperature` | 0.0 (greedy decoding) |
| Timing method | Client-side wall-clock |

**Generation tok/s** = `completion_tokens / (total_wall_time − TTFT)`

**Prompt tok/s** = `prompt_tokens / TTFT` (approximate — TTFT is wall-clock to first streaming token)

**TTFT** is measured via a separate streaming request (`max_tokens=1`), recording wall-clock time to the first chunk containing content. A fallback to the first-received data chunk handles models that return empty content deltas (e.g., gpt-oss-20b).

### Common vLLM Flags (All Models)

```
--host 0.0.0.0 --port 8000
--tensor-parallel-size 1
--max-num-seqs 64
--gpu-memory-utilization 0.9          # 0.99 for Mixtral-8x7B
--dtype auto
--attention-backend TRITON_ATTN
--mm-encoder-attn-backend TRITON_ATTN
```

### Per-Model Overrides

| Model | Extra Flags | Reason |
|---|---|---|
| LFM2-1.2B | `--enforce-eager` | SSM architecture; HIP graph capture unsupported |
| LFM2-8B-A1B | `--enforce-eager` | Same |
| gpt-oss-20b | `--trust-remote-code` | Custom `GptOssForCausalLM` architecture |
| gpt-oss-120b | `--trust-remote-code --max-model-len 8192` | KV cache budget; 70 GiB weights leave ~34 GiB |
| Mixtral-8x7B | `--enforce-eager --gpu-memory-utilization 0.99 --max-model-len 4096 --max-num-seqs 4` | Memory-constrained; still failed OOM |

---

## 8. Architecture Reference

Models benchmarked under vLLM, with architecture details drawn from the GGUF benchmark for context:

| Model | Architecture | Total Params | Active Params | vLLM Weights | GGUF File Size |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | Dense Transformer | 0.8B | 0.8B | BF16 | ~0.5 GB (Q4_K_XL) |
| LFM2-1.2B | SSM MoE (Liquid AI) | 1.2B | 1.2B | BF16 | 0.73 GB (Q4_K_S) |
| Qwen3.5-2B | Dense Transformer (GDN hybrid) | 2B | 2B | BF16 | 1.34 GB (Q4_K_XL) |
| Ministral-3-3B-BF16 | Dense Transformer (Mistral) | 3B | 3B | BF16 | 2.85 GB (Q4_K_M) |
| granite-4.0-h-tiny | Hybrid MoE attention (IBM) | ~4B | ~4B | BF16 | 4.25 GB (Q4_K_M) |
| LFM2-8B-A1B | SSM MoE (Liquid AI) | 8B | ~1B | BF16 | 4.80 GB (Q4_K_M) |
| gpt-oss-20b | Dense MXFP4 MoE (OpenAI) | 20B | 20B | MXFP4 | 12.1 GB |
| gpt-oss-120b | Dense MXFP4 MoE (OpenAI) | 120B | 120B | MXFP4 | 63.3 GB |
| Mixtral-8x7B | Dense BF16 MoE (Mistral) | 46.7B | ~13B | BF16 | N/A (not in GGUF bench) |

---

## 9. Key Observations & Recommendations

- **Best vLLM gen throughput:** LFM2-1.2B (76.6 tok/s) — despite the SSM penalty, still #1
- **Best prompt throughput:** Qwen3.5-0.8B at 706 tok/s — efficient prefill for RAG workloads
- **Best vLLM efficiency vs GGUF:** Ministral-3-3B-BF16 at −51% — BF16 full-precision minimizes format conversion overhead
- **Worst vLLM efficiency:** LFM2-8B-A1B at −85% — `--enforce-eager` eliminates all graph optimization
- **Largest model successfully loaded:** gpt-oss-120b at 70.29 GiB in unified GTT
- **Memory ceiling:** Mixtral-8x7B BF16 (~93 GB) cannot fit alongside KV cache in 104.9 GiB GTT — use quantized variants instead
- **vLLM is the right tool when:** serving multiple concurrent users, running agentic pipelines with queued requests, or needing an OpenAI-compatible production API. For single-user local chat, the GGUF stack remains faster on this hardware.

---

## License & Reuse

This report is published under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Benchmark scripts and supporting code are MIT-licensed.

If you reuse these results in publications, presentations, or product decisions, attribution is appreciated:

> Pascua, R. (2026). *Z13 Local AI Stack — vLLM Benchmark Report (Baseline: 512 MB UMA Buffer).*

This work is part of a community contribution to the [Lemonade by AMD](https://github.com/lemonade-sdk/lemonade) ecosystem and the AMD AI developer community.
