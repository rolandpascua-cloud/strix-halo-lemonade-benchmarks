# Z13 Local AI Stack — Lemonade Benchmark Report (Tuned: 96 GB Dedicated VRAM)

> **Part 2 of 4** in the *Z13 Local AI Stack Benchmark Series* — measuring the effect of allocating dedicated GPU VRAM and kernel TTM/hugepage tuning on inference performance.
>
> **Series:** [Lemonade · 512M UMA](Z13-Lemonade-Benchmark-512M.md) · [Lemonade · 96 GB VRAM *(this report)*](Z13-Lemonade-Benchmark-96G.md) · [vLLM · 512M UMA](Z13-vLLM-Benchmark-512M.md) · [vLLM · 96 GB VRAM](Z13-vLLM-Benchmark-96G.md)

| Field | Value |
|---|---|
| **Author** | Roland Pascua |
| **Platform** | ASUS ROG Flow Z13 GZ302EA · AMD Ryzen AI MAX+ 395 (Strix Halo) |
| **GPU** | Radeon 8060S (gfx1151 / RDNA 3.5 · 40 CUs · 2900 MHz boost) |
| **NPU** | AMD XDNA 2 (not tested in this run) |
| **Memory** | 128 GB LPDDR5x-8533 unified · ~273 GB/s bandwidth |
| **iGPU VRAM (BIOS)** | UMA Buffer = **96 GB dedicated** |
| **Kernel tuning** | `transparent_hugepage=always`, `numa_balancing=disabled`, `ttm.pages_limit=25165824`, `ttm.page_pool_size=25165824` |
| **OS / Kernel** | Fedora 43 · Linux 7.0.7-100.fc43.x86_64 |
| **Inference Stack** | Lemonade Server v10.4.0 · llamacpp `rocm-preview` + Vulkan |
| **Date** | 2026-05-16 |
| **Models Tested** | 38 generative (same set as baseline) |
| **Baseline** | [Z13-Lemonade-Benchmark-512M.md](Z13-Lemonade-Benchmark-512M.md) |
| **License** | Report under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · scripts under MIT |

---

## Table of Contents

1. [Configuration Change Summary](#1-configuration-change-summary)
2. [Key Findings](#2-key-findings)
3. [Full Comparison Table](#3-full-comparison-table)
4. [Analysis](#4-analysis)
5. [Updated Specialist Rankings](#5-updated-specialist-rankings)
6. [Updated Recommendations](#6-updated-recommendations)
7. [Configuration Details](#7-configuration-details)

---

## 1. Configuration Change Summary

### What Changed

| Parameter | Baseline (512M UMA) | This Run (96 GB Dedicated) |
|---|---|---|
| **iGPU VRAM (BIOS)** | Dynamic 512 MB UMA Buffer (shared from 128 GB pool) | **96 GB dedicated** |
| **System RAM available** | ~120 GB (OS sees full pool minus reservation) | **~30 GB** |
| **GPU VRAM (rocm-smi)** | 536 MB local + ~105 GiB GTT (dynamic) | **103,079,215,104 B (96 GiB dedicated)** |
| **transparent_hugepage** | default | `always` |
| **numa_balancing** | default | `disabled` |
| **ttm.pages_limit** | default | 25,165,824 pages (~96 GB) |
| **ttm.page_pool_size** | default | 25,165,824 pages (~96 GB) |

### Effect

Dedicating 96 GB as iGPU VRAM removes the dynamic unified-memory arbitration between CPU and GPU. The GPU memory manager can now use the full VRAM pool without competing with OS and CPU workloads for pages, and the TTM page pool is capped to match. Combined with `transparent_hugepage=always`, large model weight tensors are mapped using 2 MB pages instead of 4 KB pages, reducing TLB pressure during matrix multiply.

The trade-off is system RAM: only ~30 GB remains for the OS, the Lemonade server process, and any CPU-side workloads. All 38 tested models still fit in the 96 GB VRAM pool (largest: Qwen3.5-122B at 72.6 GB).

---

## 2. Key Findings

1. **Every model improved.** Generation throughput increased on all 38 models — average improvement ~10%, ranging from +1% (LFM2-8B-A1B, near VRAM bandwidth ceiling) to +22.5% (LFM2-1.2B, now the fastest model tested).

2. **LFM2-1.2B reclaims #1** at **224.7 tok/s** (was 183.5), surpassing Qwen3.5-0.8B (215.2 tok/s). The SSM architecture benefits disproportionately from dedicated memory bandwidth.

3. **Small fast models gained the most.** Models under 5 GB file size improved 12–20%: LFM2-1.2B +22.5%, Ministral-3B +19.0%, Qwen2.5-Omni-3B +19.6%, Bonsai-8B +19.8%, Llama-3.2-3B +16.7%, Qwen2.5-VL-3B +15.6%. These models are compute-bound and now benefit from fully dedicated memory bandwidth.

4. **Prompt speed (prefill) improved across the board**, including models that were previously slow at prefill. Gemma-3-4b: 487 → 561 tok/s (+15%). Qwen3.5-9B: 281 → 349 tok/s (+24%). LFM2-1.2B: 1412 → 1887 tok/s (+34%).

5. **Dense >15B wall persists, but TTFT improved significantly.** Qwen3.5-27B and Qwen3.6-27B generation speed barely changed (+1.8%), but their TTFT dropped from 0.66–0.70s to 0.50–0.51s. The kernel bottleneck is in the compute path, not memory latency.

6. **No regressions.** All models that were previously interactive remain interactive. The minimum improvement was LFM2-8B-A1B at +1.0% — it was already near the VRAM bandwidth ceiling at 153.6 tok/s.

---

## 3. Full Comparison Table

Ranked by new generation throughput. *Δgen* = new − old tok/s.

| Rank | Model | Old tok/s | **New tok/s** | Δgen | Δ% | Old prompt | New prompt | Old TTFT | New TTFT |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | **LFM2-1.2B-GGUF** | 183.5 | **224.7** | +41.2 | +22% | 1412 | 1887 | 0.03s | 0.02s |
| 2 | **Qwen3.5-0.8B-GGUF** | 191.6 | **215.2** | +23.6 | +12% | 2104 | 2384 | 0.02s | 0.02s |
| 3 | **LFM2-8B-A1B-GGUF** | 153.6 | **155.1** | +1.5 | +1% | 392 | 409 | 0.11s | 0.10s |
| 4 | **Qwen3.5-2B-GGUF** | 101.3 | **119.8** | +18.5 | +18% | 1155 | 1374 | 0.04s | 0.03s |
| 5 | **granite-4.0-h-tiny-GGUF** | 104.8 | **111.0** | +6.2 | +6% | 128 | 143 | 0.03s | 0.03s |
| 6 | **LFM2-24B-A2B-GGUF** | 93.7 | **104.6** | +10.9 | +12% | 149 | 163 | 0.28s | 0.26s |
| 7 | Qwen2.5-VL-3B-Instruct-GGUF | 78.8 | **91.1** | +12.3 | +16% | 71 | 76 | 0.01s | 0.01s |
| 8 | Qwen2.5-Omni-3B-GGUF | 75.6 | **90.4** | +14.8 | +20% | 70 | 79 | 0.01s | 0.01s |
| 9 | Bonsai-8B-gguf | 74.9 | **89.7** | +14.8 | +20% | 73 | 84 | 0.01s | 0.01s |
| 10 | SmolLM3-3B-GGUF | 75.8 | **87.8** | +12.0 | +16% | 68 | 73 | 0.01s | 0.01s |
| 11 | Llama-3.2-3B-Instruct-GGUF | 72.9 | **85.1** | +12.2 | +17% | 67 | 74 | 0.01s | 0.01s |
| 12 | Qwen3-30B-A3B-GGUF | 76.3 | **83.3** | +7.0 | +9% | 63 | 66 | 0.02s | 0.02s |
| 13 | Qwen3-Coder-30B-A3B-Instruct-GGUF | 72.3 | **78.8** | +6.5 | +9% | 57 | 67 | 0.02s | 0.01s |
| 14 | Ministral-3-3B-Instruct-2512-GGUF | 63.8 | **75.9** | +12.1 | +19% | 63 | 70 | 0.02s | 0.01s |
| 15 | Qwen3-4B-GGUF | 68.0 | **75.0** | +7.0 | +10% | 61 | 67 | 0.02s | 0.01s |
| 16 | Phi-4-mini-instruct-GGUF | 67.6 | **72.8** | +5.2 | +8% | 65 | 70 | 0.02s | 0.01s |
| 17 | gpt-oss-20b-mxfp4-GGUF | 66.1 | **70.0** | +3.9 | +6% | 113 | 121 | 0.04s | 0.04s |
| 18 | Qwen3-VL-4B-Instruct-GGUF | 60.8 | **66.7** | +5.9 | +10% | 55 | 62 | 0.02s | 0.02s |
| 19 | Gemma-3-4b-it-GGUF | 57.5 | **63.9** | +6.4 | +11% | 487 | **561** | 0.08s | 0.07s |
| 20 | Nemotron-3-Nano-30B-A3B-GGUF | 56.2 | **62.3** | +6.1 | +11% | 146 | 171 | 0.38s | 0.30s |
| 21 | GLM-4.7-Flash-GGUF | 54.3 | **60.7** | +6.4 | +12% | 43 | 51 | 0.02s | 0.02s |
| 22 | Qwen3.5-35B-A3B-GGUF | 49.5 | **53.1** | +3.6 | +7% | 183 | 193 | 0.26s | 0.25s |
| 23 | Qwen3-Next-80B-A3B-Instruct-GGUF | 47.7 | **51.5** | +3.8 | +8% | 115 | 128 | 0.40s | 0.36s |
| 24 | gpt-oss-120b-mxfp-GGUF | 47.9 | **49.8** | +1.9 | +4% | 73 | 79 | 0.07s | 0.06s |
| 25 | Gemma-4-26B-A4B-it-GGUF | 43.6 | **46.7** | +3.1 | +7% | 186 | 216 | 0.26s | 0.22s |
| 26 | Qwen3-Coder-Next-GGUF | 39.6 | **43.4** | +3.8 | +10% | 129 | 145 | 0.36s | 0.32s |
| 27 | Qwen2.5-Omni-7B-GGUF | 38.3 | **41.3** | +3.0 | +8% | 38 | 40 | 0.03s | 0.02s |
| 28 | Qwen2.5-VL-7B-Instruct-GGUF | 39.4 | **41.1** | +1.7 | +4% | 39 | 40 | 0.03s | 0.02s |
| 29 | DeepSeek-Qwen3-8B-GGUF | 38.5 | **39.5** | +1.0 | +3% | 36 | 38 | 0.03s | 0.03s |
| 30 | Qwen3-VL-8B-Instruct-GGUF | 38.1 | **39.2** | +1.1 | +3% | 37 | 38 | 0.03s | 0.03s |
| 31 | Qwen3.5-9B-GGUF | 31.2 | **33.2** | +2.0 | +6% | 281 | **349** | 0.17s | 0.14s |
| 32 | GLM-4.5-Air-UD-Q4K-XL-GGUF | 23.0 | **24.1** | +1.1 | +5% | 21 | 22 | 0.05s | 0.04s |
| 33 | Qwen3.5-122B-A10B-GGUF | 20.8 | **21.6** | +0.8 | +4% | 47 | 56 | 1.01s | 0.86s |
| 34 | Devstral-Small-2507-GGUF | 14.0 | **14.5** | +0.5 | +4% | 14 | 14 | 0.07s | 0.07s |
| 35 | Qwen3.6-27B-GGUF | 11.4 | **11.6** | +0.2 | +2% | 72 | 94 | 0.66s | **0.51s** |
| 36 | Qwen3.5-27B-GGUF | 11.3 | **11.5** | +0.2 | +2% | 69 | 95 | 0.70s | **0.50s** |
| 37 | Qwen2.5-Coder-32B-Instruct-GGUF | 10.5 | **10.9** | +0.4 | +4% | 10 | 11 | 0.10s | 0.09s |
| 38 | Gemma-4-31B-it-GGUF | 9.9 | **10.6** | +0.7 | +7% | 58 | 80 | 0.82s | **0.60s** |

---

## 4. Analysis

### Where the Gains Come From

The improvement pattern reveals two distinct GPU regimes on this platform:

**Memory-bandwidth-bound models (small, compute-light)** — Models under ~5 GB, particularly the LFM2 SSM family and sub-4B dense models, spent significant time waiting for weights to be paged in from the shared unified pool. Dedicated VRAM provides consistent, uncontested bandwidth. These models gained 12–22%.

**Compute-bound models (large MoE, MXFP4)** — Models like LFM2-8B-A1B (153.6→155.1, +1%) and gpt-oss-120b-mxfp (47.9→49.8, +4%) were already saturating the GPU compute units. Memory arbitration overhead was a smaller fraction of their runtime, so dedicated VRAM helps less.

**Dense >15B kernel wall** — The generation bottleneck for Qwen3.5-27B, Qwen3.6-27B, Qwen2.5-Coder-32B, and Gemma-4-31B is in the Vulkan shader kernel for large dense matrix multiply — not memory allocation. These models gained only 2–7% in generation speed. However, **their prefill speed improved significantly** (Qwen3.5-27B: 69→95 tok/s prompt, +37.7%; Gemma-4-31B: 58→80 tok/s, +37.9%) and **TTFT dropped substantially** (Qwen3.5-27B: 0.70s→0.50s, −28%; Qwen3.6-27B: 0.66s→0.51s, −23%). These models remain unusable for interactive generation but are now better for batch/prefill-heavy workloads.

### Ranking Changes

| Change | Model | Reason |
|---|---|---|
| #2 → #1 | LFM2-1.2B | +22.5% — SSM architecture benefits most from dedicated bandwidth |
| #1 → #2 | Qwen3.5-0.8B | +12.3% — still fastest dense model, but LFM2-1.2B outpaced it |
| #5 → #4 | Qwen3.5-2B | +18.3% — overtakes granite |
| #4 → #5 | granite-4.0-h-tiny | +5.9% — smaller relative gain |
| #8 → #12 | Qwen3-30B-A3B | +9.2% — outpaced by several smaller models gaining 15–20% |
| #11 → #9 | Bonsai-8B | +19.8% — one of the biggest gainers |
| #10 → #8 | Qwen2.5-Omni-3B | +19.6% — jumps ahead of Bonsai and SmolLM3 |

### Prompt Speed Highlights

| Model | Old prompt tok/s | New prompt tok/s | Δ% |
|---|---:|---:|---|
| LFM2-1.2B | 1412 | **1887** | +34% |
| Qwen3.5-9B | 281 | **349** | +24% |
| LFM2-8B-A1B | 392 | **409** | +4% |
| Gemma-3-4b | 487 | **561** | +15% |
| Qwen3.5-2B | 1155 | **1374** | +19% |
| Qwen3.6-27B | 72 | **94** | +31% |
| Qwen3.5-27B | 69 | **95** | +38% |
| Gemma-4-31B | 58 | **80** | +38% |
| GLM-4.7-Flash | 43 | **51** | +19% |
| Gemma-4-26B-A4B | 186 | **216** | +16% |

The prompt speed improvement is consistent and often larger than the generation improvement. Prefill is more parallelizable (batched matrix-vector products) and benefits more from dedicated memory bandwidth than generation (which is autoregressive — one token at a time).

### Dense Wall: Still Broken, but TTFT Meaningfully Improved

| Model | Gen old→new | TTFT old→new |
|---|---|---|
| Qwen3.5-27B | 11.3→11.5 (+2%) | 0.70s → 0.50s (−29%) |
| Qwen3.6-27B | 11.4→11.6 (+2%) | 0.66s → 0.51s (−23%) |
| Gemma-4-31B | 9.9→10.6 (+7%) | 0.82s → 0.60s (−27%) |
| Qwen3.5-122B-A10B | 20.8→21.6 (+4%) | 1.01s → 0.86s (−15%) |

Generation throughput for these models is still kernel-limited (not memory-limited), so the wall persists. But TTFT — which includes prompt evaluation — improved substantially because prefill benefits from dedicated VRAM. For streaming use cases where the first token matters, these models are meaningfully more responsive.

---

## 5. Updated Specialist Rankings

### Speed (generation throughput)

| Rank | Model | New tok/s | Old tok/s | Δ |
|---:|---|---:|---:|---|
| 1 | LFM2-1.2B | **224.7** | 183.5 | +41.2 |
| 2 | Qwen3.5-0.8B | **215.2** | 191.6 | +23.6 |
| 3 | LFM2-8B-A1B | **155.1** | 153.6 | +1.5 |
| 4 | Qwen3.5-2B | **119.8** | 101.3 | +18.5 |
| 5 | granite-4.0-h-tiny | **111.0** | 104.8 | +6.2 |
| 6 | LFM2-24B-A2B | **104.6** | 93.7 | +10.9 |

### Long-context / RAG (prompt processing speed)

| Rank | Model | New prompt tok/s | Old prompt tok/s |
|---:|---|---:|---:|
| 1 | Qwen3.5-0.8B | **2384** | 2104 |
| 2 | LFM2-1.2B | **1887** | 1412 |
| 3 | Qwen3.5-2B | **1374** | 1155 |
| 4 | Gemma-3-4b | **561** | 487 |
| 5 | LFM2-8B-A1B | **409** | 392 |
| — | Qwen3.5-9B | **349** | 281 *(notable outlier — see baseline §3)* |

### Multimodal (vision/audio capable)

| Model | New tok/s | Old tok/s | Δ% |
|---|---:|---:|---|
| Qwen2.5-VL-3B | **91.1** | 78.8 | +16% |
| Qwen2.5-Omni-3B | **90.4** | 75.6 | +20% |
| Qwen3-VL-4B | **66.7** | 60.8 | +10% |
| Gemma-3-4b | **63.9** | 57.5 | +11% |
| Gemma-4-26B-A4B | **46.7** | 43.6 | +7% |
| Qwen2.5-Omni-7B | **41.3** | 38.3 | +8% |
| Qwen2.5-VL-7B | **41.1** | 39.4 | +4% |
| Qwen3-VL-8B | **39.2** | 38.1 | +3% |

---

## 6. Updated Recommendations

### Three-Model Core Stack (Updated Speeds)

| Role | Model | New speed | Old speed |
|---|---|---|---|
| **Primary assistant** | Qwen3-30B-A3B-GGUF | 83.3 tok/s | 76.3 tok/s |
| **Speed tier** | LFM2-8B-A1B-GGUF | 155.1 tok/s | 153.6 tok/s |
| **Coding** | Qwen3-Coder-30B-A3B | 78.8 tok/s | 72.3 tok/s |

### Updated "Models to Add" Speeds

| Need | Model | New speed |
|---|---|---|
| Ultra-fast routing | Qwen3.5-0.8B | 215.2 tok/s |
| Speed tier (alt) | LFM2-1.2B | 224.7 tok/s |
| Speed + quality | LFM2-24B-A2B | 104.6 tok/s |
| Max coding quality | Qwen3-Coder-Next | 43.4 tok/s |
| Vision (fastest) | Qwen2.5-VL-3B | 91.1 tok/s |
| Vision + audio | Qwen2.5-Omni-3B | 90.4 tok/s |
| Long RAG prefill | Gemma-3-4b | 63.9 tok/s (561 prompt) |

### Models to Avoid (unchanged from baseline)

The dense >15B wall models (Qwen3.5-27B, Qwen3.6-27B, Qwen2.5-Coder-32B, Gemma-4-31B, Devstral-22B) remain unsuitable for interactive generation. The VRAM change improved their TTFT and prefill speed but did not meaningfully change their generation throughput. See the [baseline report's recommendations](Z13-Lemonade-Benchmark-512M.md#models-to-avoid-in-current-configuration) for alternatives.

### Operational Trade-Off

Choose the 96 GB dedicated VRAM configuration if your workload is primarily GPU inference and you can live with ~30 GB of system RAM. Stay on the 512 MB UMA buffer if you need:
- More than ~30 GB available for CPU workloads (large dataset preprocessing, browser tabs, IDEs)
- Flexibility to occasionally exceed 96 GB for GPU (e.g., to keep both gpt-oss-120b and Qwen3.5-122B resident simultaneously)

For most local AI developer workflows, the 96 GB configuration is a net win — the 10% average inference speedup and prefill improvements outweigh the lost system RAM, and the system remains responsive at 30 GB for typical desktop use.

---

## 7. Configuration Details

### Hardware State During This Run

| Component | Value |
|---|---|
| Platform | ASUS ROG Flow Z13 GZ302EA |
| CPU | AMD Ryzen AI MAX+ 395 (16c/32t) |
| System RAM | ~30 GB (OS-visible, post-VRAM carve-out) |
| GPU VRAM | 96 GiB dedicated (Radeon 8060S, gfx1151, RDNA 3.5) |
| NPU | XDNA 2 (not tested in this run) |

### Active Kernel Parameters

```
transparent_hugepage=always
numa_balancing=disable
ttm.pages_limit=25165824
ttm.page_pool_size=25165824
```

> **Note:** `amdttm.pages_limit` (originally specified in deployment intent) was not present in `/proc/cmdline` — this parameter may not exist in the running kernel. `ttm.pages_limit` is the effective parameter. All other parameters are confirmed active.

### Inference Stack

| Component | Version |
|---|---|
| Lemonade Server | v10.4.0 (snap rev 177) |
| GPU backend | llamacpp `rocm-preview` + Vulkan (gfx1151) |
| ROCm | 7.12.0 (bundled in snap) |
| OS / Kernel | Fedora 43 / 7.0.7-100.fc43.x86_64 |

### Benchmark Parameters

Identical to baseline run: same prompt, same `max_tokens=256`, same `temperature=0`, same 1 warm-up + 3 measured runs, server-side timings. `/no_thinking` appended for Qwen3 models. See [baseline methodology §8](Z13-Lemonade-Benchmark-512M.md#8-benchmark-mechanics) for full details.

---

## License & Reuse

This report is published under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Benchmark scripts and supporting code are MIT-licensed.

If you reuse these results in publications, presentations, or product decisions, attribution is appreciated:

> Pascua, R. (2026). *Z13 Local AI Stack — Lemonade Benchmark Report (Tuned: 96 GB Dedicated VRAM).*

This work is part of a community contribution to the [Lemonade by AMD](https://github.com/lemonade-sdk/lemonade) ecosystem and the AMD AI developer community.
