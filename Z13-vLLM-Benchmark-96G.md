# Z13 Local AI Stack — vLLM Benchmark Report (Tuned: 96 GB Dedicated VRAM)

> **Part 4 of 4** in the *Z13 Local AI Stack Benchmark Series* — vLLM serving on AMD Strix Halo under dedicated 96 GB VRAM, focused on the Qwen3.5 hybrid Mamba+Attention architecture.
>
> **Series:** [Lemonade · 512M UMA](Z13-Lemonade-Benchmark-512M.md) · [Lemonade · 96 GB VRAM](Z13-Lemonade-Benchmark-96G.md) · [vLLM · 512M UMA](Z13-vLLM-Benchmark-512M.md) · [vLLM · 96 GB VRAM *(this report)*](Z13-vLLM-Benchmark-96G.md)

| Field | Value |
|---|---|
| **Author** | Roland Pascua |
| **Platform** | ASUS ROG Flow Z13 GZ302EA · AMD Ryzen AI MAX+ 395 (Strix Halo) |
| **GPU** | Radeon 8060S (gfx1151 / RDNA 3.5 · 40 CUs · 2900 MHz boost) |
| **Memory** | 128 GB LPDDR5x-8533 unified · ~273 GB/s bandwidth |
| **iGPU VRAM (BIOS)** | UMA Buffer = **96 GB dedicated** |
| **OS / Kernel** | Fedora 43 · Linux 7.0.7-100.fc43.x86_64 |
| **vLLM Version** | v0.20.1+rocm721 (ROCm, bundled in Lemonade snap) |
| **Precision** | bfloat16 (no quantization) |
| **Date** | 2026-05-16 |
| **Models Tested** | 2 (Qwen3.5-4B, Qwen3.5-9B) |
| **License** | Report under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · scripts under MIT |

---

## 1. Summary

| Model | Gen tok/s | TTFT | Precision | vs llamacpp baseline |
|---|---:|---:|---|---|
| **Qwen3.5-4B-vLLM** | **24.3** | 0.07s | bfloat16 | 3.1× slower than Qwen3-4B-GGUF (75.0) |
| **Qwen3.5-9B-vLLM** | **12.2** | 0.17s | bfloat16 | 2.7× slower than Qwen3.5-9B-GGUF (33.2) |

**Bottom line:** vLLM single-stream throughput on gfx1151 is 2.7–3.1× lower than llamacpp with Q4_K_M quantization. vLLM is designed for batch/concurrent inference, not single-stream latency. For interactive use on this platform, llamacpp via Lemonade remains the better backend.

---

## 2. vLLM Configuration

### Launch Method

vLLM is bundled inside the Lemonade snap, but for these models **it cannot be used via the Lemonade proxy** — Lemonade does not pass `--max-num-seqs` to vllm-server, causing vLLM to fail during Mamba cache block initialization for Qwen3.5's hybrid Mamba+Attention architecture. As a workaround, the server was launched directly:

```bash
/var/snap/lemonade-server/common/cache/lemonade/bin/vllm/rocm/bin/vllm-server \
  --model <model_path> \
  --served-model-name <name> \
  --port 18000 \
  --max-model-len 8192 \
  --gpu-memory-utilization <util> \
  --max-num-seqs 64
```

| Parameter | 4B | 9B |
|---|---|---|
| `--gpu-memory-utilization` | 0.25 | 0.45 |
| `--max-num-seqs` | 64 | 64 |
| `--max-model-len` | 8192 | 8192 |
| Precision | bfloat16 (default) | bfloat16 (default) |
| Port | 18000 | 18000 |

**Why `--max-num-seqs 64`:** Qwen3.5 uses a hybrid Mamba+Attention architecture. vLLM requires dedicated Mamba cache blocks for each sequence slot. With the default `--max-num-seqs 1024`, vLLM fails with:

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (658)
```

Reducing to 64 resolves this. This affects batch capacity but not single-stream throughput.

### Model Paths (HuggingFace cache)

```
/var/snap/lemonade-server/common/.cache/huggingface/hub/
  models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

These are the unquantized HuggingFace weights used by the Lemonade vLLM integration. They differ from the GGUF-quantized weights used by the llamacpp benchmarks.

---

## 3. Benchmark Methodology

### Why Server-Side Timings Are Unavailable

The llamacpp benchmarks use server-side `timings.predicted_per_second` and `timings.prompt_per_second` from the response body. vLLM's OpenAI-compatible `/v1/chat/completions` endpoint does not include a `timings` block. All measurements use **client-side wall-clock timing**:

- **Gen tok/s:** `completion_tokens / total_elapsed` (non-streaming, full response)
- **TTFT:** time from request send to first content chunk (streaming call)
- **Prompt tok/s:** not measurable with client-side timing — reported as 0

### Warm-up and Run Count

Same parameters as the llamacpp benchmarks: 1 warm-up + 3 measured runs. Median of the 3 runs is recorded.

### Per-Run Server Restart (ROCm Stability)

On gfx1151, the bundled vLLM ROCm build crashes after 3–4 inference requests. The benchmark script (`bench_vllm.py`) works around this by **killing and restarting vllm-server before every run** — warm-up and each of the 3 measured runs each get a fresh server start. Startup takes ~30–60 seconds per restart, so the total benchmark runtime is approximately:

```
(4 restarts × ~45s) + (4 × inference time) ≈ 6–8 minutes per model
```

This is a known stability limitation of the ROCm build on this hardware revision.

---

## 4. Detailed Results

### Qwen3.5-4B-vLLM

| Run | TTFT | Gen tok/s | Tokens |
|---|---:|---:|---:|
| Warm-up | — | 19.1 | 256 |
| Run 1 | 0.07s | 24.2 | 256 |
| Run 2 | 0.09s | 24.3 | 256 |
| Run 3 | 0.07s | 24.5 | 256 |
| **Median** | **0.07s** | **24.3** | 256 |

### Qwen3.5-9B-vLLM

| Run | TTFT | Gen tok/s | Tokens |
|---|---:|---:|---:|
| Warm-up | — | 10.8 | 256 |
| Run 1 | 0.16s | 12.4 | 256 |
| Run 2 | 0.18s | 12.1 | 256 |
| Run 3 | 0.17s | 12.2 | 256 |
| **Median** | **0.17s** | **12.2** | 256 |

---

## 5. vLLM vs llamacpp Comparison

### Important Caveats

The comparison is not perfectly apples-to-apples:

1. **Quantization:** llamacpp runs Q4_K_M (4-bit, ~55% of original weight size). vLLM runs bfloat16 (16-bit, full precision). Lower quantization = faster matrix multiply, less memory bandwidth pressure.
2. **Model generation:** Qwen3-4B-GGUF (llamacpp) is Qwen3 generation; Qwen3.5-4B-vLLM is Qwen3.5 generation. Different models.
3. **Backend architecture:** llamacpp is optimized for single-stream autoregressive decode via Vulkan compute shaders on gfx1151. vLLM is a production serving engine optimized for batch concurrency and uses custom CUDA/HIP kernels.

Despite these caveats, the comparison is useful for deciding which backend to deploy for interactive single-user inference.

### Generation Throughput

| Model (llamacpp) | tok/s | Model (vLLM) | tok/s | vLLM vs llamacpp |
|---|---:|---|---:|---:|
| Qwen3-4B-GGUF (Q4_K_M) | 75.0 | Qwen3.5-4B-vLLM (bf16) | 24.3 | **3.1× slower** |
| Qwen3.5-9B-GGUF (Q4_K_M) | 33.2 | Qwen3.5-9B-vLLM (bf16) | 12.2 | **2.7× slower** |

### TTFT

| Model (llamacpp) | TTFT | Model (vLLM) | TTFT | Comparison |
|---|---:|---|---:|---|
| Qwen3-4B-GGUF | 0.01s | Qwen3.5-4B-vLLM | 0.07s | 7× slower |
| Qwen3.5-9B-GGUF | 0.14s | Qwen3.5-9B-vLLM | 0.17s | 1.2× slower |

TTFT for the 4B model is 7× worse under vLLM. For the 9B model the gap narrows to 1.2×, likely because the 9B model occupies more VRAM and its prefill benefits more from vLLM's batched attention kernels.

---

## 6. Analysis

### Why vLLM Is Slower for Single-Stream Inference Here

vLLM's performance advantage over llamacpp materializes under concurrent load (multiple users/requests). For single-stream inference:

- **llamacpp Vulkan path** on gfx1151 is highly optimized for Q4_K_M quantized weights. The quantization alone reduces memory bandwidth demand by ~4×.
- **vLLM's ROCm HIP kernels** are designed for CUDA-class datacenter GPUs (A100, H100, MI300X). On gfx1151 (RDNA 3.5 APU), the kernel implementations are not tuned for this microarchitecture and the overhead of the production serving infrastructure (paged KV cache, scheduler, chunked prefill) dominates at single-stream.
- **bfloat16 memory bandwidth:** At 24.3 tok/s with a 4B parameter model at bf16, the GPU is moving 4B × 2 bytes = 8 GB of weights per token, requiring ~195 GB/s effective bandwidth — the Radeon 8060S can theoretically deliver ~273 GB/s but achieves far less due to overhead in the vLLM kernel path.

### When vLLM Would Be Preferred

- **High concurrency:** If serving multiple simultaneous users, vLLM's continuous batching scheduler can serve N requests at nearly the same throughput as 1 request in llamacpp, while llamacpp queues them serially.
- **Large model + full precision:** Applications requiring full bfloat16 precision (avoiding quantization artifacts) for 4B or 9B class models.
- **Structured output / function calling:** vLLM's guided decoding support is more robust than llamacpp's grammar-based approach for complex schemas.

For all single-user interactive inference on this hardware, **llamacpp via Lemonade is the correct backend.**

---

## 7. Known Issues and Limitations

### ROCm gfx1151 Stability

The bundled vLLM ROCm build crashes after 3–4 requests. This is a known instability on gfx1151 (RDNA 3.5 with integrated memory controller). The crash manifests as a silent process exit (no Python traceback). The per-run server restart workaround is effective but adds significant benchmark overhead and would not be acceptable in production.

### Lemonade vLLM Proxy Incompatibility — Bug Report

Lemonade's built-in vLLM proxy (`lemon run --backend vllm`) does not pass `--max-num-seqs` to vllm-server. This causes an immediate startup failure for any model using the Mamba architecture (Qwen3.5 series). Until Lemonade adds this parameter to its vLLM launch configuration, Qwen3.5 models cannot be served through the Lemonade proxy with vLLM.

**Reproduction:**
1. Configure Lemonade with vLLM backend.
2. Attempt to load any Qwen3.5 model.
3. Observe failure: `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (N)`.

**Suggested fix:** Add a configurable `--max-num-seqs` flag to the Lemonade vLLM launch configuration, defaulting to a value compatible with Mamba/hybrid architectures (e.g., 64).

### Kernel Parameters (Pending Reboot)

The updated kernel parameters specified for this run:

```
ttm.pages_limit=27648000
amdttm.pages_limit=27648000
```

were **not yet active** during this benchmark. The running `/proc/cmdline` still showed the previous values:

```
ttm.pages_limit=25165824
ttm.page_pool_size=25165824
```

The new values require regeneration of the systemd-boot BLS entry (via `grubby`) and a reboot. The impact on vLLM performance is expected to be small — the bottleneck is compute, not memory page management — but the [VRAM-96 llamacpp benchmarks](Z13-Lemonade-Benchmark-96G.md) were also run with the old values, so the comparison is consistent.

---

## 8. Configuration Details

### Hardware State During This Run

| Component | Value |
|---|---|
| Platform | ASUS ROG Flow Z13 GZ302EA |
| CPU | AMD Ryzen AI MAX+ 395 (16c/32t) |
| System RAM | ~30 GB (OS-visible, 96 GiB carved for GPU) |
| GPU | Radeon 8060S, gfx1151, RDNA 3.5 |
| GPU VRAM | 96 GiB dedicated |
| NPU | XDNA 2 (not used) |

### Inference Stack

| Component | Value |
|---|---|
| vLLM version | v0.20.1+rocm721 |
| ROCm version | 7.2.1 (bundled in Lemonade snap) |
| vLLM binary | `/var/snap/lemonade-server/common/cache/lemonade/bin/vllm/rocm/bin/vllm-server` |
| API endpoint | `http://localhost:18000/v1` |
| Lemonade version | v10.4.0 (snap rev 177) — vLLM launched directly, not via proxy |
| OS / Kernel | Fedora 43 / 7.0.7-100.fc43.x86_64 |
| Results file | `~/benchmark_results_vllm.json` |

### Benchmark Script

`bench_vllm.py` — restarts vllm-server before each run; uses `/v1/chat/completions` (non-streaming for throughput, streaming for TTFT); client-side wall-clock timing; identical prompt and token budget to llamacpp benchmarks.

Benchmark prompt:

> "Explain the difference between supervised and unsupervised machine learning. Cover how each approach works, typical use cases, and the key trade-offs. Be thorough but concise."

`/no_thinking` appended for Qwen3.x model names (consistent with llamacpp benchmarks).

`max_tokens=256`, `temperature=0.0`, 1 warm-up + 3 measured runs, median reported.

---

## License & Reuse

This report is published under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Benchmark scripts and supporting code are MIT-licensed.

If you reuse these results in publications, presentations, or product decisions, attribution is appreciated:

> Pascua, R. (2026). *Z13 Local AI Stack — vLLM Benchmark Report (Tuned: 96 GB Dedicated VRAM).*

This work is part of a community contribution to the [Lemonade by AMD](https://github.com/lemonade-sdk/lemonade) ecosystem and the AMD AI developer community.
