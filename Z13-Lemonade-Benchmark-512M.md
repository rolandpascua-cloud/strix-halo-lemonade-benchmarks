# Z13 Local AI Stack — Lemonade Benchmark Report (Baseline: 512 MB UMA Buffer)

> **Part 1 of 4** in the *Z13 Local AI Stack Benchmark Series* — a community evaluation of Lemonade Server and adjacent inference stacks on AMD Strix Halo silicon.
>
> **Series:** [Lemonade · 512M UMA *(this report)*](Z13-Lemonade-Benchmark-512M.md) · [Lemonade · 96 GB VRAM](Z13-Lemonade-Benchmark-96G.md) · [vLLM · 512M UMA](Z13-vLLM-Benchmark-512M.md) · [vLLM · 96 GB VRAM](Z13-vLLM-Benchmark-96G.md)

| Field | Value |
|---|---|
| **Author** | Roland Pascua |
| **Platform** | ASUS ROG Flow Z13 GZ302EA · AMD Ryzen AI MAX+ 395 (Strix Halo) |
| **GPU** | Radeon 8060S (gfx1151 / RDNA 3.5 · 40 CUs · 2900 MHz boost) |
| **NPU** | AMD XDNA 2 (Phoenix/Strix Halo generation, 8 spatial columns) |
| **Memory** | 128 GB LPDDR5x-8533 unified · ~273 GB/s bandwidth |
| **iGPU VRAM (BIOS)** | UMA Buffer = **512 MB** (dynamic GTT allocation up to ~105 GiB) |
| **OS / Kernel** | Fedora 43 · Linux 7.0.7-100.fc43.x86_64 |
| **Inference Stack** | Lemonade Server v10.4.0 · llamacpp `rocm-preview` + Vulkan · FastFlowLM v0.9.40 (NPU) |
| **Date** | 2026-05-15 |
| **Models Tested** | 38 generative + 2 utility |
| **License** | Report under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · scripts under MIT |

---

## Table of Contents

1. [Executive Summary & Rankings](#1-executive-summary--rankings)
2. [Model Size & Architecture Overview](#2-model-size--architecture-overview)
3. [Deep-Dive Performance Analysis](#3-deep-dive-performance-analysis)
4. [Individual Model Profiles](#4-individual-model-profiles)
5. [Strategic Recommendations](#5-strategic-recommendations--utility-mapping)
6. [Hardware Context](#6-hardware-context)
7. [Software & Environment](#7-software--environment-context)
8. [Benchmark Methodology](#8-benchmark-mechanics)
9. [Known Issues & Bug Reports](#9-known-issues--bug-reports)

---

## 1. Executive Summary & Rankings

### Overview

Thirty-eight generative models and two utility models (embedding, reranking) were benchmarked on a single-node local inference stack with **no cloud dependency**. With the BIOS UMA Buffer set to 512 MB, GPU memory is allocated dynamically from the 128 GB unified pool via GTT, allowing every tested model — up to 480B parameters — to load without compromise.

Three findings dominate the results:

1. **Generation throughput is governed by quantization format and architecture, not parameter count.** Models using MXFP4 or highly-sparse MoE architectures run 3–10× faster than equivalently-sized dense models in Q4_K_M format on the Radeon 8060S Vulkan backend.

2. **The LFM2 SSM architecture is the fastest inference architecture tested by a wide margin.** Liquid AI's SSM Mixture-of-Experts design (not Transformer-based) delivers throughput no Transformer can match at equivalent active parameter count. All three LFM2 variants rank in the top 6 by generation speed.

3. **Dense Q4_K_M models above ~15B are universally unusable on the Vulkan backend.** This wall is confirmed across three generations (Qwen3.5, Qwen3.6, Qwen2.5-Coder) and six distinct models: Devstral-22B (14 tok/s), Gemma-4-31B (9.9), Qwen3.5-27B (11.3), Qwen3.6-27B (11.4), Qwen2.5-Coder-32B (10.5). No dense Q4_K_M model above 15B escaped this ceiling.

The NPU (XDNA 2) underperforms the GPU for text generation in all head-to-head comparisons under the current FastFlowLM 0.9.40 stack. The bundled vLLM backend (v0.20.1-rocm7.12.0) is **non-operational** through the Lemonade proxy — all model loads return HTTP 500 (see [Known Issues](#9-known-issues--bug-reports) and the companion [vLLM benchmark report](Z13-vLLM-Benchmark-512M.md), which uses a direct-launch container as a workaround).

### Overall Leaderboard (Generation Speed)

Ranked by median generation throughput across 3 measured runs.

| Rank | Model | Params (Active) | Size | Gen tok/s | Prompt tok/s | TTFT | Architecture |
|---:|---|---|---|---:|---:|---|---|
| 1 | **Qwen3.5-0.8B-GGUF** | 0.8B | ~0.5 GB | **191.6** | 2104 | 0.02s | Dense |
| 2 | **LFM2-1.2B-GGUF** | 1.2B | 0.73 GB | **183.5** | 1412 | 0.03s | SSM MoE |
| 3 | **LFM2-8B-A1B-GGUF** | 8B (~1B) | 4.80 GB | **153.6** | 392 | 0.11s | SSM MoE |
| 4 | **granite-4.0-h-tiny-GGUF** | ~4B | 4.25 GB | **104.8** | 128 | 0.03s | Dense |
| 5 | **Qwen3.5-2B-GGUF** | 2B | 1.34 GB | **101.3** | 1155 | 0.04s | Dense |
| 6 | **LFM2-24B-A2B-GGUF** | 24B (~2B) | 14.40 GB | **93.7** | 149 | 0.28s | SSM MoE |
| 7 | Qwen2.5-VL-3B-Instruct-GGUF | 3B+vision | ~2.2 GB | 78.8 | 71 | 0.01s | Dense+Vision |
| 8 | Qwen3-30B-A3B-GGUF | 30B (~3B) | ~14 GB | 76.3 | 63 | 0.02s | MoE |
| 9 | SmolLM3-3B-GGUF | 3B | 1.94 GB | 75.8 | 68 | 0.01s | Dense |
| 10 | Qwen2.5-Omni-3B-GGUF | 3B+vision | 4.6 GB | 75.6 | 70 | 0.01s | Multimodal |
| 11 | Bonsai-8B-gguf | 8B | 1.2 GB | 74.9 | 73 | 0.01s | Dense |
| 12 | Llama-3.2-3B-Instruct-GGUF | 3B | ~2.0 GB | 72.9 | 67 | 0.01s | Dense |
| 13 | Qwen3-Coder-30B-A3B-Instruct-GGUF | 30B (~3B) | ~16 GB | 72.3 | 57 | 0.02s | MoE |
| 14 | Qwen3-4B-GGUF | 4B | ~2.4 GB | 68.0 | 61 | 0.02s | Dense |
| 15 | Phi-4-mini-instruct-GGUF | ~3.8B | 2.49 GB | 67.6 | 65 | 0.02s | Dense |
| 16 | gpt-oss-20b-mxfp4-GGUF | 20B | 12.1 GB | 66.1 | 113 | 0.04s | Dense MXFP4 |
| 17 | Ministral-3-3B-Instruct-2512-GGUF | 3B | 2.85 GB | 63.8 | 63 | 0.02s | Dense |
| 18 | Qwen3-VL-4B-Instruct-GGUF | 4B | 3.33 GB | 60.8 | 55 | 0.02s | Dense+Vision |
| 19 | Gemma-3-4b-it-GGUF | 4B | 3.61 GB | 57.5 | **487** | 0.08s | Dense+Vision |
| 20 | Nemotron-3-Nano-30B-A3B-GGUF | 30B (~3B) | 22.8 GB | 56.2 | 146 | 0.38s | MoE |
| 21 | GLM-4.7-Flash-GGUF | ~16B | 17.60 GB | 54.3 | 43 | 0.02s | Dense |
| 22 | Qwen3.5-35B-A3B-GGUF | 35B (~3B) | 19.70 GB | 49.5 | 183 | 0.26s | MoE |
| 23 | gpt-oss-120b-mxfp-GGUF | 120B | 63.3 GB | 47.9 | 73 | 0.07s | Dense MXFP4 |
| 24 | Qwen3-Next-80B-A3B-Instruct-GGUF | 80B (~3B) | 45.10 GB | 47.7 | 115 | 0.40s | MoE |
| 25 | Gemma-4-26B-A4B-it-GGUF | 26B (~4B) | 16.9 GB | 43.6 | 186 | 0.26s | MoE+Vision |
| 26 | Qwen3-Coder-Next-GGUF | ~480B (~35B) | 43.7 GB | 39.6 | 129 | 0.36s | MoE MXFP4 |
| 27 | Qwen2.5-VL-7B-Instruct-GGUF | 7B+vision | ~4.8 GB | 39.4 | 39 | 0.03s | Dense+Vision |
| 28 | DeepSeek-Qwen3-8B-GGUF | 8B | 5.25 GB | 38.5 | 36 | 0.03s | Dense |
| 29 | Qwen2.5-Omni-7B-GGUF | 7B+vision | 7.2 GB | 38.3 | 38 | 0.03s | Multimodal |
| 30 | Qwen3-VL-8B-Instruct-GGUF | 8B+vision | ~5.2 GB | 38.1 | 37 | 0.03s | Dense+Vision |
| 31 | Qwen3.5-9B-GGUF | 9B | 5.97 GB | 31.2 | 281 | 0.17s | Dense |
| 32 | GLM-4.5-Air-UD-Q4K-XL-GGUF | Large MoE | ~73 GB | 23.0 | 21 | 0.05s | MoE |
| 33 | Qwen3.5-122B-A10B-GGUF | 122B (~10B) | 72.6 GB | 20.8 | 47 | 1.01s | MoE |
| 34 | Devstral-Small-2507-GGUF | ~22B | 14.3 GB | 14.0 | 14 | 0.07s | Dense |
| 35 | Qwen3.6-27B-GGUF | 27B | ~16 GB | 11.4 | 72 | 0.66s | Dense ❌ |
| 36 | Qwen3.5-27B-GGUF | 27B | ~16 GB | 11.3 | 69 | 0.70s | Dense ❌ |
| 37 | Qwen2.5-Coder-32B-Instruct-GGUF | 32B | ~18 GB | 10.5 | 10 | 0.10s | Dense ❌ |
| 38 | Gemma-4-31B-it-GGUF | 31B | 18.3 GB | 9.9 | 58 | 0.82s | Dense ❌ |

> ❌ = Dense Q4_K_M >15B hits the Vulkan kernel wall — unusable for interactive inference. See §3 for analysis.

### Specialist Rankings

**Speed (interactive responsiveness)**

| Rank | Model | Gen tok/s | TTFT |
|---:|---|---:|---|
| 1 | Qwen3.5-0.8B | 191.6 | 0.02s |
| 2 | LFM2-1.2B | 183.5 | 0.03s |
| 3 | LFM2-8B-A1B | 153.6 | 0.11s |
| 4 | granite-4.0-h-tiny | 104.8 | 0.03s |
| 5 | Qwen3.5-2B | 101.3 | 0.04s |

**Long-context / RAG (prompt processing speed)**

| Rank | Model | Prompt tok/s | Notes |
|---:|---|---:|---|
| 1 | Qwen3.5-0.8B | 2104 | Processes 128k context in ~61s |
| 2 | LFM2-1.2B | 1412 | Processes 128k context in ~91s |
| 3 | Qwen3.5-2B | 1155 | Processes 128k context in ~111s |
| 4 | Gemma-3-4b | 487 | Outstanding prefill for 4B class |
| 5 | LFM2-8B-A1B | 392 | Fast prefill at 8B scale |

**Quality/Scale (largest models at interactive speed, ≥35 tok/s)**

| Rank | Model | Params | Gen tok/s |
|---:|---|---|---:|
| 1 | gpt-oss-120b-mxfp | 120B | 47.9 |
| 2 | Qwen3-Next-80B-A3B | 80B (~3B active) | 47.7 |
| 3 | Qwen3-Coder-Next | ~480B (~35B active) | 39.6 |
| 4 | Qwen3.5-35B-A3B | 35B (~3B active) | 49.5 |
| 5 | Qwen3-30B-A3B | 30B (~3B active) | 76.3 |

**Multimodal (vision/audio capable)**

| Model | Modalities | Gen tok/s |
|---|---|---:|
| Qwen2.5-VL-3B | Text + Vision | 78.8 |
| Qwen2.5-Omni-3B | Text + Vision + Audio | 75.6 |
| Qwen3-VL-4B | Text + Vision | 60.8 |
| Gemma-3-4b | Text + Vision | 57.5 |
| Gemma-4-26B-A4B | Text + Vision | 43.6 |
| Qwen2.5-VL-7B | Text + Vision | 39.4 |
| Qwen2.5-Omni-7B | Text + Vision + Audio | 38.3 |
| Qwen3-VL-8B | Text + Vision | 38.1 |

**Utility (non-generative)**

| Model | Type | Throughput |
|---|---|---|
| nomic-embed-text-v1 | Embeddings (768-dim) | 540 docs/s |
| bge-reranker-v2-m3 | Cross-encoder reranking | 6.0 ms/doc (5-doc) · 4.8 ms/doc (20-doc) |

---

## 2. Model Size & Architecture Overview

### The Quantization Divide

The single largest performance predictor on this platform is quantization format and architecture class. The Radeon 8060S Vulkan backend has optimized kernels for MXFP4 and MoE sparse operations. Dense Q4_K_M models larger than ~15B suffer severe throughput penalties. The LFM2 SSM architecture additionally bypasses Transformer attention bottlenecks entirely.

| Format | Example | Expected throughput | Observed |
|---|---|---|---|
| SSM MoE (LFM2) | LFM2-8B-A1B | Exceptional | 153.6 tok/s ✅ |
| MXFP4 dense | gpt-oss-20b-mxfp4 | High | 66.1 tok/s ✅ |
| MXFP4 MoE | Qwen3-Coder-Next | Moderate (large model) | 39.6 tok/s ✅ |
| Q4_K_M MoE | Qwen3-30B-A3B | High (few active params) | 76.3 tok/s ✅ |
| Q4_K_M dense <10B | Bonsai-8B, Llama-3B | High | 63–75 tok/s ✅ |
| Q4_K_M dense 10–15B | GLM-4.7-Flash (~16B) | Moderate | 54.3 tok/s ⚠️ |
| Q4_K_M dense >15B | Qwen3.5-27B, Qwen3.6-27B, Qwen2.5-Coder-32B, Devstral-22B, Gemma-4-31B | Should be moderate | **9–14 tok/s ❌** |
| UD-Q4_K_XL MoE | Qwen3.5-35B-A3B, Qwen3-Next-80B | Moderate | 47–50 tok/s ✅ |

The dense >15B wall is now confirmed across **five distinct models and three architecture generations**. It is not a fluke — it is a structural limitation of the Vulkan backend's kernel implementation for large dense weight matrices. MXFP4 is the only viable quantization for dense models above this threshold.

> **Note:** Qwen3.5 MoE models (35B, 122B) use UD-Q4_K_XL quantization but run slower than Qwen3-generation MoE models at similar active param counts, suggesting architectural changes in Qwen3.5 that are less efficient on the Vulkan backend.

### Architecture & Size Reference

| Model | Architecture | Total Params | Active Params | File Size | Quantization |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | Dense | 0.8B | 0.8B | ~0.5 GB | Q4_K_XL |
| LFM2-1.2B | SSM MoE (Liquid AI) | 1.2B | 1.2B | 0.73 GB | Q4_K_S |
| Qwen3.5-2B | Dense | 2B | 2B | 1.34 GB | Q4_K_XL |
| SmolLM3-3B | Dense | 3B | 3B | 1.94 GB | Q4_K_M |
| Phi-4-mini | Dense | ~3.8B | ~3.8B | 2.49 GB | Q4_K_M |
| Ministral-3B | Dense | 3B | 3B | 2.85 GB | Q4_K_M |
| Llama-3.2-3B | Dense | 3B | 3B | ~2.0 GB | Q4_K_M |
| Qwen2.5-VL-3B | Dense+Vision | 3B+enc | 3B | ~2.2 GB | Q4_K_M |
| Qwen3-4B | Dense | 4B | 4B | ~2.4 GB | Q4_K_M |
| Qwen3-VL-4B | Dense+Vision | 4B | 4B | 3.33 GB | Q4_K_M |
| Gemma-3-4b | Dense+Vision | 4B | 4B | 3.61 GB | Q4_K_M |
| granite-4.0-h-tiny | Dense | ~4B | ~4B | 4.25 GB | Q4_K_M |
| LFM2-8B-A1B | SSM MoE (Liquid AI) | 8B | ~1B | 4.80 GB | Q4_K_M |
| Bonsai-8B | Dense | 8B | 8B | 1.2 GB | Ultra-compressed |
| Qwen2.5-Omni-3B | Multimodal MoE | 3B+enc | 3B | 4.6 GB total | Q4_K_M |
| DeepSeek-Qwen3-8B | Dense | 8B | 8B | 5.25 GB | Q4_K_M |
| Qwen3.5-9B | Dense | 9B | 9B | 5.97 GB | Q4_K_M |
| Qwen2.5-VL-7B | Dense+Vision | 7B+enc | 7B | ~4.8 GB | Q4_K_M |
| Qwen3-VL-8B | Dense+Vision | 8B+enc | 8B | ~5.2 GB | Q4_K_M |
| Qwen2.5-Omni-7B | Multimodal | 7B+enc | 7B | 7.2 GB total | Q4_K_M |
| gpt-oss-20b-mxfp4 | Dense | 20B | 20B | 12.1 GB | MXFP4 |
| LFM2-24B-A2B | SSM MoE (Liquid AI) | 24B | ~2B | 14.40 GB | Q4_K_M |
| Devstral-Small | Dense | ~22B | ~22B | 14.3 GB | Q4_K_M |
| Qwen3-30B-A3B | MoE | 30B | ~3B | ~14 GB | Q4_K_M |
| Qwen3-Coder-30B-A3B | MoE | 30B | ~3B | ~16 GB | Q4_K_M |
| Gemma-4-26B-A4B | MoE+Vision | 26B | ~4B | 16.9 GB | Q4_K_M |
| GLM-4.7-Flash | Dense | ~16B | ~16B | 17.60 GB | Q4_K_M |
| Qwen3.5-35B-A3B | MoE | 35B | ~3B | 19.70 GB | UD-Q4_K_XL |
| Gemma-4-31B | Dense | 31B | 31B | 18.3 GB | Q4_K_M |
| Qwen3.5-27B | Dense | 27B | 27B | ~16 GB | Q4_K_M |
| Qwen3.6-27B | Dense | 27B | 27B | ~16 GB | Q4_K_M |
| Qwen2.5-Coder-32B | Dense | 32B | 32B | ~18 GB | Q4_K_M |
| Nemotron-3-Nano-30B | MoE | 30B | ~3B | 22.8 GB | Q4_K_M |
| Qwen3-Next-80B-A3B | MoE | 80B | ~3B | 45.10 GB | UD-Q4_K_XL |
| Qwen3-Coder-Next | MoE | ~480B | ~35B | 43.7 GB | MXFP4 |
| Qwen3.5-122B-A10B | MoE | 122B | ~10B | 72.6 GB | UD-Q4_K_XL |
| GLM-4.5-Air | MoE (Zhipu AI) | Large | ~Large | ~73 GB | UD-Q4_K_XL |
| gpt-oss-120b-mxfp | Dense | 120B | 120B | 63.3 GB | MXFP4 |

---

## 3. Deep-Dive Performance Analysis

### Generation Throughput

The 128 GB unified memory pool means no model tested required disk swapping. Throughput bands emerge cleanly by architecture class:

| Band | Range | Models | Driver |
|---|---|---|---|
| Exceptional | >100 tok/s | Qwen3.5-0.8B (191.6), LFM2-1.2B (183.5), LFM2-8B-A1B (153.6), granite-tiny (104.8), Qwen3.5-2B (101.3) | LFM2 SSM MoE; small dense sub-1B; granite hybrid |
| Near-exceptional | 80–100 tok/s | LFM2-24B-A2B (93.7), Qwen2.5-VL-3B (78.8), Qwen3-30B-A3B (76.3) | LFM2 large; fast multimodal; efficient MoE |
| Fast | 60–80 tok/s | SmolLM3 (75.8), Omni-3B (75.6), Bonsai-8B (74.9), Llama-3B (72.9), Qwen3-Coder-30B (72.3), Qwen3-4B (68.0), Phi-4-mini (67.6), gpt-oss-20b (66.1), Ministral-3B (63.8), Qwen3-VL-4B (60.8) | Efficient quantization or low active param count |
| Moderate | 35–60 tok/s | Gemma-3-4b (57.5), Nemotron-30B (56.2), GLM-4.7-Flash (54.3), Qwen3.5-35B (49.5), gpt-oss-120b (47.9), Qwen3-Next-80B (47.7), Gemma-4-26B (43.6), Qwen3-Coder-Next (39.6), Qwen2.5-VL-7B (39.4), DeepSeek-8B (38.5), Omni-7B (38.3), Qwen3-VL-8B (38.1) | Large total params or suboptimal kernel path |
| Slow | 20–35 tok/s | Qwen3.5-9B (31.2), GLM-4.5-Air (23.0), Qwen3.5-122B (20.8) | Architecture inefficiency at 9B; large MoE overhead |
| Unusable interactive | <20 tok/s | Devstral-22B (14.0), Qwen3.6-27B (11.4), Qwen3.5-27B (11.3), Qwen2.5-Coder-32B (10.5), Gemma-4-31B (9.9) | Dense Q4_K_M >15B on Vulkan — confirmed kernel wall |

### Prompt Evaluation (Prefill) Speed

Prompt eval speed determines how fast the model processes input before generating its first token — critical for RAG pipelines, document QA, and long-system-prompt setups.

| Tier | Prompt tok/s | Models |
|---|---|---|
| Exceptional (>500) | 2104, 1412, 1155, 487, 392 | Qwen3.5-0.8B, LFM2-1.2B, Qwen3.5-2B, Gemma-3-4b, LFM2-8B-A1B |
| High (100–400) | 113–281 | Qwen3.5-9B (281), Gemma-4-26B (186), Qwen3.5-35B (183), LFM2-24B (149), Nemotron-30B (146), Qwen3-Next-80B (115), Qwen3-Coder-Next (129), gpt-oss-20b (113) |
| Standard (50–100) | 55–73 | Most 3–8B dense models and MoE models with moderate routing |
| Poor (<50) | 10–47 | GLM-4.7-Flash (43), Qwen3.5-122B (47), Devstral (14), DeepSeek-8B (36), Omni-7B (38), Qwen3-VL-8B (37), Qwen2.5-VL-7B (39), Qwen2.5-Coder-32B (10), GLM-4.5-Air (21) |

Qwen3.5-9B's 281 tok/s prompt speed is a notable outlier — despite slow generation (31.2 tok/s), its prefill is among the fastest dense models tested, making it a niche RAG candidate where inputs are long and outputs are short.

### Latency (TTFT — Time to First Token)

| TTFT Range | Models |
|---|---|
| <0.05s (instant) | Qwen3.5-0.8B, Qwen2.5-VL-3B, SmolLM3, Bonsai-8B, Llama-3B, Qwen2.5-Omni-3B, Qwen3-30B-A3B, Qwen3-4B, Phi-4-mini, Ministral-3B, Qwen3-Coder-30B, Qwen3-VL-4B, GLM-4.7-Flash, GLM-4.5-Air |
| 0.05–0.15s (fast) | LFM2-1.2B (0.03s), granite (0.03s), Qwen3.5-2B (0.04s), gpt-oss-20b (0.04s), gpt-oss-120b (0.07s), Devstral (0.07s), Gemma-3-4b (0.08s), Qwen2.5-Coder-32B (0.10s), LFM2-8B-A1B (0.11s) |
| 0.15–0.45s (noticeable) | Qwen3.5-9B (0.17s), Qwen3.5-35B (0.26s), Gemma-4-26B (0.26s), LFM2-24B (0.28s), Qwen3-Coder-Next (0.36s), Nemotron-30B (0.38s), Qwen3-Next-80B (0.40s) |
| >0.60s (slow) | Qwen3.5-27B (0.70s), Qwen3.6-27B (0.66s), Gemma-4-31B (0.82s), Qwen3.5-122B (1.01s) |

### NPU vs GPU Head-to-Head

| Model Family | GPU Backend | GPU tok/s | NPU Backend | NPU tok/s | GPU Advantage |
|---|---|---:|---|---:|---:|
| Llama-3.2-3B | llamacpp:rocm | 72.9 | FastFlowLM 0.9.40 | 22.1 | **3.3×** |
| Qwen3-4B | llamacpp:rocm | 68.0 | FastFlowLM 0.9.40 | 17.6 | **3.9×** |

NPU TTFT is also 47–61× higher (0.95–1.22s vs 0.01–0.02s). The XDNA 2 NPU's advantage in this stack is zero-GPU-contention background inference, not throughput.

---

## 4. Individual Model Profiles

### Qwen3.5-0.8B-GGUF
- **Best Use Cases:** Ultra-fast classification, intent detection, routing, autocomplete, high-volume batch processing where even LFM2-1.2B's 183 tok/s is a bottleneck
- **Positives:** New overall #1 at **191.6 tok/s**. **2104 tok/s prompt speed** — new record by a wide margin (processes 128k context in ~61s). Near-instant 0.02s TTFT. Smallest viable model for real-time production workloads.
- **Negatives:** 0.8B parameter ceiling makes this unsuitable for complex reasoning, factual recall, or instruction following. A routing/classification model, not a general assistant. Pair with a larger model in tiered architectures.

### LFM2-1.2B-GGUF
- **Best Use Cases:** Classification, intent detection, short summarization, routing, real-time autocomplete, high-volume batch processing
- **Positives:** Second fastest at 183.5 tok/s. 1412 tok/s prompt speed. Sub-0.03s TTFT. Liquid AI's SSM architecture (non-Transformer) gives it disproportionate speed. Tiny footprint (730 MB).
- **Negatives:** 1.2B parameters limits reasoning depth, factual recall, and instruction complexity. Not suitable as a primary reasoning model.

### LFM2-8B-A1B-GGUF
- **Best Use Cases:** Fast general-purpose inference requiring more knowledge than the sub-2B models; rapid summarization; batch API workloads where speed and quality must balance
- **Positives:** 153.6 tok/s — third fastest model tested overall. 392 tok/s prompt speed. 8B total parameters with only ~1B active via MoE routing gives it significantly more factual breadth than smaller models at minimal speed cost. 0.11s TTFT.
- **Negatives:** MoE routing means not all 8B parameters contribute to each token — quality ceiling is closer to a ~1-2B dense model than a full 8B model. Limited community evaluation vs Qwen/Llama alternatives.

### LFM2-24B-A2B-GGUF
- **Best Use Cases:** High-throughput tasks requiring genuine 20B+ knowledge depth; the best single model for users who want both speed and quality without compromise
- **Positives:** 93.7 tok/s at 24B total / ~2B active — sixth fastest model tested. 149 tok/s prompt speed. The speed-quality trade-off is the best in the benchmark.
- **Negatives:** 0.28s TTFT (noticeable for single-turn chat). 14.4 GB footprint. MoE active-param ceiling (~2B) means complex multi-step reasoning may still lag behind dense 7B+ competitors.

### granite-4.0-h-tiny-GGUF
- **Best Use Cases:** IBM enterprise NLP, structured output generation, agentic tool-calling, code assistance at scale
- **Positives:** 104.8 tok/s — fourth fastest overall, outpacing all standard Transformer models. IBM's thinking-capable architecture delivers structured reasoning at tiny cost. Strong instruction following.
- **Negatives:** IBM-focused training data may limit breadth. Smaller knowledge base than 7B+ alternatives.

### Qwen3.5-2B-GGUF
- **Best Use Cases:** Fast drafting, rapid iteration, background task processing, edge-like workloads
- **Positives:** 101.3 tok/s gen, 1155 tok/s prompt. Thinking-capable at 2B scale. Excellent TTFT (0.04s). Qwen3.5 architecture improvements over Qwen3.
- **Negatives:** 2B parameter ceiling limits output quality on complex reasoning tasks.

### Qwen3-30B-A3B-GGUF
- **Best Use Cases:** Daily general-purpose chat, writing, analysis, reasoning, long-form content
- **Positives:** 76.3 tok/s with 30B total parameters. The MoE architecture (~3B active) explains the speed paradox. Thinking mode available. 0.02s TTFT. Best Transformer-family quality-speed ratio tested.
- **Negatives:** Prompt prefill speed (63 tok/s) is moderate. Not multimodal.

### Qwen3-Coder-30B-A3B-Instruct-GGUF
- **Best Use Cases:** Code generation, debugging, refactoring, technical documentation, code review
- **Positives:** 72.3 tok/s at 30B total params. Purpose-trained for code. Tool-calling capable. Thinking mode for complex multi-step code problems.
- **Negatives:** General knowledge sacrificed for code specialization. Prompt prefill (57 tok/s) is the lowest of the Qwen3 MoE models.

### Qwen3-Coder-Next-GGUF (480B MoE)
- **Best Use Cases:** Most complex coding tasks, architecture design, multi-file refactoring, research-grade code analysis
- **Positives:** Largest model tested at ~480B parameters (~35B active). MXFP4 quantization keeps it at 39.6 tok/s. Highest knowledge ceiling of any coding model tested.
- **Negatives:** 0.36s TTFT is noticeable. 43.7 GB. For most tasks, Qwen3-Coder-30B at 72 tok/s produces comparable results in half the time.

### Qwen3-Next-80B-A3B-Instruct-GGUF
- **Best Use Cases:** Tasks demanding large general knowledge at interactive speed; complex reasoning; high-quality writing and analysis where the 30B models reach their ceiling
- **Positives:** 47.7 tok/s at 80B total / ~3B active. Excellent MoE efficiency via UD-Q4_K_XL quantization. Tool-calling capable. Strong balance of size and usability.
- **Negatives:** 0.40s TTFT. 45.1 GB. Qwen3-30B-A3B runs 60% faster at similar active param count — use Qwen3-Next-80B only when the additional knowledge depth is needed.

### Qwen3-VL-4B-Instruct-GGUF
- **Best Use Cases:** Vision tasks at 4B scale — image captioning, visual QA, document OCR, multimodal agents
- **Positives:** 60.8 tok/s with full vision capability. Only a ~10% speed penalty vs the text-only Qwen3-4B (68.0 tok/s). 0.02s TTFT. Compact at 3.33 GB.
- **Negatives:** Lower prompt throughput than Gemma-3-4b (55 vs 487 tok/s) — not ideal for long vision-document pipelines.

### Qwen3.5-35B-A3B-GGUF
- **Best Use Cases:** Tasks requiring large model knowledge with tool-calling; situations where Qwen3.5 architecture is specifically needed
- **Positives:** 49.5 tok/s at 35B total / ~3B active. Tool-calling capable. Strong 183 tok/s prompt speed. UD-Q4_K_XL quantization.
- **Negatives:** Runs slower than Qwen3-30B-A3B (76.3 tok/s) at nearly the same active parameter count — the Qwen3.5 architecture changes are less efficient on this Vulkan backend.

### Qwen3.5-9B-GGUF
- **Best Use Cases:** Long-context RAG where inputs are very long but outputs are short; document QA pipelines where 281 tok/s prefill matters more than generation speed
- **Positives:** Best-in-class prompt speed for a dense 9B model (281 tok/s). Tool-calling capable. Thinking mode available.
- **Negatives:** **31.2 tok/s generation** — unusually slow for a 9B dense model (cf. Bonsai-8B at 74.9 tok/s). Not suitable as a primary conversational model.

### Qwen3.5-122B-A10B-GGUF
- **Best Use Cases:** Maximum Qwen3.5 quality when generation speed is secondary; batch processing; complex tasks where 10B active params provide noticeably better reasoning than 3B-active MoE alternatives
- **Positives:** Largest Qwen3.5 model tested. ~10B active parameters provide significantly more reasoning capacity than the 3B-active MoE siblings. Tool-calling capable.
- **Negatives:** **20.8 tok/s** — at the boundary of interactive usability. **1.01s TTFT** is the highest of any model tested.

### Qwen3.5-27B-GGUF
- **Best Use Cases:** ⚠️ Not recommended for interactive use
- **Negatives:** **11.3 tok/s** — hits the dense Q4_K_M >15B Vulkan wall. 0.70s TTFT. Use Qwen3.5-35B-A3B (MoE, 49.5 tok/s) instead, or Qwen3-30B-A3B (76.3 tok/s).

### Qwen3.6-27B-GGUF
- **Best Use Cases:** ⚠️ Not recommended for interactive use
- **Positives:** Qwen3.6 architecture — presumably improvements over Qwen3.5. 72 tok/s prompt speed (the wall only affects generation).
- **Negatives:** **11.4 tok/s** — same dense wall as Qwen3.5-27B. The generation bottleneck is architecture-agnostic. 0.66s TTFT.

### Qwen2.5-Coder-32B-Instruct-GGUF
- **Best Use Cases:** ⚠️ Not recommended — use Qwen3-Coder-30B-A3B instead
- **Negatives:** **10.5 tok/s** — dense 32B Q4_K_M hits the hardest point of the Vulkan wall. Qwen3-Coder-30B-A3B at 72.3 tok/s delivers the same code-specialized capability at 6.9× the speed via MoE architecture.

### Qwen2.5-VL-3B-Instruct-GGUF
- **Best Use Cases:** Fastest multimodal inference — image captioning, visual QA, document parsing
- **Positives:** **78.8 tok/s** — fastest multimodal model tested. 71 tok/s prompt speed. Sub-0.01s TTFT. Compact ~2.2 GB footprint.
- **Negatives:** Text + vision only (no audio unlike Omni-3B). 3B parameter ceiling for vision reasoning.

### Qwen2.5-VL-7B-Instruct-GGUF
- **Best Use Cases:** Higher-quality vision inference where Qwen2.5-VL-3B quality is insufficient
- **Positives:** 7B parameters provides more visual reasoning depth than the 3B variant.
- **Negatives:** 39.4 tok/s — exactly half the speed of Qwen2.5-VL-3B (78.8 tok/s).

### Qwen3-VL-8B-Instruct-GGUF
- **Best Use Cases:** Vision tasks requiring the full 8B parameter budget with Qwen3 architecture; vision + thinking mode
- **Positives:** 8B dense + vision. Qwen3 architecture brings thinking-mode capability to visual reasoning tasks. 0.03s TTFT.
- **Negatives:** 38.1 tok/s — similar to Qwen2.5-VL-7B; the VL-3B at 78.8 tok/s is twice as fast.

### GLM-4.5-Air-UD-Q4K-XL-GGUF
- **Best Use Cases:** Zhipu AI ecosystem; large MoE inference; tasks where GLM training data provides an advantage
- **Positives:** 23.0 tok/s despite very large file size (~73 GB) — MoE architecture prevents it from hitting the dense wall. 0.05s TTFT.
- **Negatives:** 73 GB download and footprint for 23.0 tok/s generation is poor value — Qwen3.5-122B-A10B (72.6 GB) delivers comparable speed at 20.8 tok/s with better-documented architecture.

### GLM-4.7-Flash-GGUF
- **Best Use Cases:** Zhipu AI ecosystem integrations; tool-calling workflows
- **Positives:** 54.3 tok/s at ~16B dense — reasonable throughput for its size. 0.02s TTFT. Tool-calling capable.
- **Negatives:** Prompt throughput (43 tok/s) is among the lowest of the mid-range models.

### Qwen3-4B-GGUF
- **Best Use Cases:** Fast general chat, lightweight agents
- **Positives:** 68.0 tok/s, thinking-capable. Solid instruction following at 4B scale. 0.02s TTFT.
- **Negatives:** Outperformed on speed by MoE 30B models while offering less knowledge depth.

### Qwen2.5-Omni-3B-GGUF
- **Best Use Cases:** Multimodal applications combining text, image, and audio. Voice transcription + response. Visual QA.
- **Positives:** 75.6 tok/s — second fastest multimodal model tested. Handles audio natively (unique in this benchmark). 0.01s TTFT.
- **Negatives:** mmproj file adds 2.5 GB. Audio/vision processing requires multimodal API path. Qwen2.5-VL-3B is slightly faster for vision-only tasks.

### Qwen2.5-Omni-7B-GGUF
- **Best Use Cases:** Higher-quality multimodal inference where Omni-3B quality is insufficient and audio capability is required
- **Negatives:** 38.3 tok/s — half the speed of Omni-3B. The 3B variant is the better choice in almost every scenario.

### gpt-oss-20b-mxfp4-GGUF
- **Best Use Cases:** General reasoning, structured output, tool-calling with >10B knowledge depth
- **Positives:** 66.1 tok/s at 20B — MXFP4 working as intended on gfx1151. Strong prompt speed (113 tok/s). Reasoning capable.
- **Negatives:** 12.1 GB footprint. No multimodal.

### gpt-oss-120b-mxfp-GGUF
- **Best Use Cases:** Maximum reasoning depth at interactive speed — complex multi-step analysis, document synthesis, agent orchestration
- **Positives:** 47.9 tok/s at 120B — best quality-at-interactive-speed for a dense model. MXFP4 is the only reason this is viable.
- **Negatives:** 63.3 GB download. For most tasks, Qwen3-30B-A3B at 76 tok/s with 30B knowledge is more practical.

### Bonsai-8B-gguf
- **Best Use Cases:** Efficient general-purpose inference at 8B quality with minimal storage
- **Positives:** 74.9 tok/s at 8B. Only 1.2 GB file despite 8B parameters. 0.01s TTFT.
- **Negatives:** Aggressive quantization may reduce output quality. Limited documentation.

### Phi-4-mini-instruct-GGUF
- **Best Use Cases:** Reasoning-heavy tasks in a small footprint, STEM QA, structured outputs
- **Positives:** 67.6 tok/s. Microsoft's Phi-4 training produces strong reasoning above its weight class.
- **Negatives:** Narrow training distribution may underperform on creative or domain-specific tasks.

### SmolLM3-3B-GGUF
- **Best Use Cases:** On-device simulation, rapid prototyping, low-memory deployment
- **Positives:** 75.8 tok/s — fastest 3B text-only model tested. Strong general instruction following.
- **Negatives:** 3B parameter ceiling. Less battle-tested than Llama/Qwen alternatives.

### Llama-3.2-3B-Instruct-GGUF
- **Best Use Cases:** General-purpose chat, well-understood baseline, broad compatibility
- **Positives:** 72.9 tok/s. Meta's training quality and ecosystem support. Predictable behavior.
- **Negatives:** No reasoning/thinking mode.

### Ministral-3B-Instruct-2512-GGUF
- **Best Use Cases:** Mistral-ecosystem workflows, API-compatible agent systems
- **Positives:** 63.8 tok/s. Good for structured JSON and tool-calling.
- **Negatives:** Slowest of the 3B text-only models. No thinking mode.

### Gemma-3-4b-it-GGUF
- **Best Use Cases:** RAG pipelines with long documents; summarization of large corpora
- **Positives:** **487 tok/s prompt speed** — best-in-class for ingesting long contexts among models above 1B. Vision-capable.
- **Negatives:** 57.5 tok/s generation is below the 3B models. Higher TTFT (0.08s).

### Gemma-4-26B-A4B-it-GGUF
- **Best Use Cases:** Multimodal analysis (images + text), visual reasoning with 26B knowledge
- **Positives:** MoE ~4B active. 186 tok/s prompt speed. Full vision capability.
- **Negatives:** 43.6 tok/s generation. 0.26s TTFT. Vision encoder adds overhead.

### Gemma-4-31B-it-GGUF
- **Best Use Cases:** ⚠️ Not recommended for interactive use
- **Negatives:** **9.9 tok/s**. Dense Q4_K_M at 31B hits the Vulkan kernel wall. 0.82s TTFT. Gemma-4-26B-A4B (MoE) runs 4.4× faster.

### Nemotron-3-Nano-30B-A3B-GGUF
- **Best Use Cases:** NVIDIA-ecosystem reasoning tasks, thinking-mode workflows
- **Positives:** 56.2 tok/s with thinking mode. 30B total / ~3B active. 146 tok/s prompt speed.
- **Negatives:** Slower than Qwen3-30B-A3B (76 tok/s) at the same MoE tier. Larger footprint (22.8 GB vs ~14 GB).

### DeepSeek-Qwen3-8B-GGUF
- **Best Use Cases:** Tasks requiring DeepSeek's distilled reasoning in compact form
- **Positives:** Thinking-capable. Strong reasoning via DeepSeek distillation.
- **Negatives:** 38.5 tok/s — significantly slower than expected for 8B dense.

### Devstral-Small-2507-GGUF
- **Best Use Cases:** ⚠️ Not recommended in current form
- **Negatives:** **14.0 tok/s** — effectively unusable for interactive coding. Qwen3-Coder-30B-A3B at 72 tok/s is the replacement.

### nomic-embed-text-v1-GGUF
- **Best Use Cases:** Semantic search indexing, RAG vector store population, document clustering
- **Positives:** 540 docs/s throughput, 768-dimensional embeddings. Essentially zero operational cost.
- **Negatives:** 768 dimensions smaller than some alternatives. Text-only.

### bge-reranker-v2-m3-GGUF
- **Best Use Cases:** RAG pipeline second-pass reranking, search result quality improvement
- **Positives:** 6.0 ms/doc at 5-doc batch, 4.8 ms/doc at 20-doc batch. Correct cross-encoder scoring.
- **Negatives:** Lemonade proxy does not forward `/v1/rerank` — must access underlying llama-server port directly (see [Known Issues](#9-known-issues--bug-reports)).

---

## 5. Strategic Recommendations & Utility Mapping

### Primary Deployment Recommendations

**Three-model core stack** covering 95% of daily workloads:

| Role | Model | Rationale |
|---|---|---|
| **Primary assistant** | Qwen3-30B-A3B-GGUF | Best Transformer quality-speed balance. 30B knowledge, 76 tok/s, thinking mode. |
| **Speed tier** | LFM2-8B-A1B-GGUF | 153.6 tok/s with 8B knowledge — replaces Qwen3.5-2B for most speed-tier tasks |
| **Coding** | Qwen3-Coder-30B-A3B-Instruct-GGUF | Purpose-trained, 72 tok/s, thinking mode for complex code |

**Add based on specific needs:**

| Need | Add | Why |
|---|---|---|
| Speed + quality beyond 8B | LFM2-24B-A2B-GGUF | 93.7 tok/s at 24B total — sixth fastest model overall |
| Maximum coding quality | Qwen3-Coder-Next-GGUF | 480B knowledge, 39.6 tok/s for complex code analysis |
| Large general knowledge | Qwen3-Next-80B-A3B-Instruct | 80B knowledge at 47.7 tok/s — best general MoE beyond 30B |
| Highest reasoning ceiling | gpt-oss-120b-mxfp | 120B dense at 47.9 tok/s via MXFP4 |
| Multimodal (vision only) | Qwen2.5-VL-3B-GGUF | 78.8 tok/s — fastest vision model tested, 3B scale |
| Multimodal (images + audio) | Qwen2.5-Omni-3B-GGUF | 75.6 tok/s — only model with native audio |
| Vision at 4B scale | Qwen3-VL-4B-Instruct-GGUF | 60.8 tok/s, 3.33 GB, thinking mode for visual reasoning |
| Long document RAG | Gemma-3-4b-it-GGUF | 487 tok/s prompt — fastest document ingestion above 1B |
| Fast RAG with 9B knowledge | Qwen3.5-9B-GGUF | 281 tok/s prompt speed — unusual prefill/generation split |
| Ultra-fast routing/classification | Qwen3.5-0.8B-GGUF | 191.6 tok/s, 2104 tok/s prompt — fastest model on any dimension |
| Semantic search | nomic-embed-text-v1 | 540 docs/s, always-on |
| Search reranking | bge-reranker-v2-m3 | 5ms/doc, correct relevance ordering |

### Models to Avoid in Current Configuration

| Model | Issue | Alternative |
|---|---|---|
| Gemma-4-31B-it | 9.9 tok/s — dense Q4_K_M >30B on Vulkan | Gemma-4-26B-A4B (MoE, 43.6 tok/s) |
| Qwen3.5-27B | 11.3 tok/s — dense 27B Q4_K_M hits wall | Qwen3.5-35B-A3B (MoE, 49.5 tok/s) or Qwen3-30B-A3B (76.3 tok/s) |
| Qwen3.6-27B | 11.4 tok/s — same wall, new architecture doesn't help | Qwen3-30B-A3B (76.3 tok/s) |
| Qwen2.5-Coder-32B | 10.5 tok/s — dense 32B coder hits wall hardest | Qwen3-Coder-30B-A3B (72.3 tok/s, MoE) |
| Devstral-Small-2507 | 14 tok/s — same root cause | Qwen3-Coder-30B-A3B (72.3 tok/s) |
| Qwen3.5-9B | 31.2 tok/s generation — architecture bottleneck | Bonsai-8B (74.9) or LFM2-8B-A1B (153.6); keep only for prefill-heavy RAG |
| Qwen3.5-122B-A10B | 20.8 tok/s, 1.01s TTFT | gpt-oss-120b-mxfp (47.9 tok/s) |
| Qwen3.5-35B-A3B | 49.5 tok/s — Qwen3.5 MoE ~35% slower than Qwen3 MoE | Qwen3-30B-A3B (76.3 tok/s) |
| DeepSeek-Qwen3-8B | 38.5 tok/s — underperforms for 8B dense | Bonsai-8B (74.9 tok/s) or LFM2-8B-A1B (153.6) |
| Qwen2.5-Omni-7B | 38.3 tok/s — half speed of Omni-3B | Qwen2.5-Omni-3B (75.6 tok/s) |
| GLM-4.5-Air | 23.0 tok/s, 73 GB — poor tok/s per GB | Qwen3-Next-80B-A3B (47.7 tok/s, 45 GB) |

### Deployment Matrix

| Task | Primary Model | Fallback / Speed Tier |
|---|---|---|
| General chat / Q&A | Qwen3-30B-A3B | LFM2-24B-A2B |
| Long-form writing | Qwen3-30B-A3B | gpt-oss-120b |
| Code generation | Qwen3-Coder-30B-A3B | Qwen3-Coder-Next (complex) |
| Code review / architecture | Qwen3-Coder-Next | Qwen3-Next-80B-A3B |
| Document QA (RAG) | Gemma-3-4b (prefill) + nomic-embed + bge-reranker | Qwen3.5-9B (prefill speed) |
| Image / visual analysis | Qwen2.5-VL-3B | Qwen3-VL-4B |
| Audio transcription + response | Qwen2.5-Omni-3B | Whisper-Large-v3-Turbo + Qwen3-30B |
| Classification / routing | Qwen3.5-0.8B | LFM2-1.2B |
| Batch summarization | LFM2-8B-A1B | LFM2-24B-A2B |
| High-throughput API | Qwen3.5-0.8B (191.6 tok/s) | LFM2-8B-A1B (153.6 tok/s) |
| Reasoning / analysis | gpt-oss-120b or Qwen3-30B-A3B (thinking) | Qwen3-Next-80B-A3B |
| Background NPU inference | llama3.2:3b (FLM) | qwen3:4b (FLM) |
| Semantic search indexing | nomic-embed-text-v1 | — |
| Search result reranking | bge-reranker-v2-m3 | — |

### Architectural Decision Rules

When evaluating new models for this platform, apply this decision filter before downloading:

1. **LFM2 SSM MoE?** → Expect exceptional speed (>100 tok/s for sub-2B active params).
2. **MXFP4 or Transformer MoE with <5B active?** → Likely viable at any total size.
3. **Qwen3.5 MoE?** → Benchmark first. Qwen3.5 runs ~35% slower than Qwen3 MoE at equivalent active params on this Vulkan backend.
4. **Q4_K_M dense with <10B params?** → Viable (60–75 tok/s typical).
5. **Q4_K_M dense with >15B params?** → Do not download. The wall is confirmed across 5 distinct models spanning three Qwen generations and Gemma. Only MXFP4 or MoE variants are viable at this size.
6. **New Qwen3.5/3.6 dense models?** → The generation penalty applies regardless of architecture improvements. Qwen3.6-27B (11.4 tok/s) confirms this is a kernel-level limitation, not a model quality issue.

---

## 6. Hardware Context

All benchmarks were collected on the following hardware, running unmodified — no overclocking, no power-limit changes, no thermal tuning.

### Host System

| Component | Specification |
|---|---|
| **Platform** | ASUS ROG Flow Z13 GZ302EA (2-in-1 tablet/laptop) |
| **CPU** | AMD Ryzen AI MAX+ 395 (Strix Halo) |
| **CPU cores** | 16 cores / 32 threads (Zen 5 architecture) |
| **CPU frequency** | Base ~2.0 GHz · Boost up to 5187 MHz |
| **CPU cache** | 64 MB L3 (unified) |
| **System RAM** | 128 GB LPDDR5x unified memory (shared pool: CPU + GPU + NPU) |
| **Memory bandwidth** | ~273 GB/s (LPDDR5x-8533 ×256-bit) |
| **System tuning** | `tuned-adm` active profile: `accelerator-performance` |
| **BIOS UMA Buffer** | **512 MB** (dynamic GTT allocation up to ~105 GiB) |

### GPU (Integrated, Radeon 8060S)

| Attribute | Value |
|---|---|
| **GPU model** | AMD Radeon 8060S |
| **GPU codename** | gfx1151 (RDNA 3.5 architecture) |
| **Compute Units** | 40 CUs |
| **Boost clock** | 2900 MHz |

`rocm-smi` reported memory state during this run:

```
VRAM Total Memory (B):      536,870,912    (~512 MiB — UMA carve-out)
VIS_VRAM Total Memory (B):  536,870,912    (visible to GPU's local heap)
GTT  Total Memory (B):  113,246,208,000    (~105 GiB — dynamically allocatable from unified pool)
```

### NPU (XDNA 2)

| Attribute | Value |
|---|---|
| **NPU type** | AMD XDNA 2 (Phoenix/Strix Halo generation) |
| **Device node** | `/dev/accel/accel0` |
| **Driver** | `amdxdna` 0.6.0 |
| **Firmware** | `17f0_11/npu_7.sbin` |
| **Columns (spatial tiles)** | 8 |
| **Inference stack** | FastFlowLM v0.9.40 (COPR package) |
| **GPU vs NPU** | GPU is 3.3–3.9× faster for text generation in every head-to-head comparison |

---

## 7. Software & Environment Context

### Operating System

| Component | Version / Detail |
|---|---|
| **OS** | Fedora 43 (KDE Plasma desktop) |
| **Kernel** | 7.0.7-100.fc43.x86_64 (pinned; not the rolling default) |
| **Bootloader** | systemd-boot (not GRUB; disk is NVMe with Btrfs root) |
| **Python (system)** | 3.14.4 |

### Inference Stack

| Component | Version | Notes |
|---|---|---|
| **Lemonade Server** | v10.4.0 (snap rev 177) | Installed via snap (`lemonade-server`); serves at port 13305 |
| **API endpoint** | `/api/v1` | OpenAI-compatible `chat/completions` |
| **GPU backend** | llamacpp `rocm-preview` + `vulkan` | Two separate llamacpp builds bundled for gfx1151 |
| **ROCm** | 7.12.0 | Bundled inside snap via `therock-dist-linux-gfx1151`; not a system-level ROCm install |
| **FastFlowLM** | v0.9.40 | COPR package; NPU inference at port 8002 (`/v1`) |
| **vLLM** | v0.20.1-rocm7.12.0 | Bundled in snap but **non-operational via Lemonade proxy** — see [Known Issues](#9-known-issues--bug-reports). Direct-launch results are in the companion [vLLM 512M report](Z13-vLLM-Benchmark-512M.md). |

### Network & Access

All inference was local (loopback). Benchmarks were run over SSH from a WSL2 client (`192.168.0.73`). No external network traffic was involved in any inference call.

---

## 8. Benchmark Mechanics

### Methodology Overview

Each model was tested with a single standardized prompt, run once for warm-up (discarded) and then three times for measurement. The median of the three measured runs is reported. All runs are sequential — no parallelism.

### Prompt

```
Explain the difference between machine learning and deep learning in exactly one sentence.
```

For any model whose name contains `Qwen3` (case-insensitive), the string `/no_thinking` is appended to the prompt to suppress extended chain-of-thought reasoning and prevent thinking-mode tokens from inflating or deflating throughput measurements.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `max_tokens` | 256 | Long enough for a complete response; short enough to measure throughput, not runtime |
| `temperature` | 0.0 | Greedy decoding — eliminates sampling variance between runs |
| `top_p` | 1.0 | No nucleus sampling |
| `stream` | False (throughput) / True (TTFT) | Two separate calls per run |

### Throughput Measurement

Throughput is measured using **server-side timings** extracted from the non-streaming API response body:

- `timings.predicted_per_second` → Generation tok/s
- `timings.prompt_eval_count` / `timings.prompt_eval_duration` → Prompt tok/s

Server-side timings reflect pure model execution time — they exclude Python overhead, HTTP serialization, and network latency. This is more accurate than client-side wall-clock measurement for comparing model and backend efficiency.

### TTFT (Time to First Token)

A separate streaming API call is made to the same endpoint. TTFT is measured as the elapsed wall-clock time from sending the HTTP request to receiving the first streaming chunk containing a generated token. This measures the combined latency of HTTP connection setup + prompt evaluation + first token generation. Streaming TTFT is the metric most relevant to interactive feel.

### API Endpoints

GPU benchmark calls go to the Lemonade Server OpenAI-compatible API:

```
http://localhost:13305/api/v1/chat/completions
```

NPU benchmark calls go to FastFlowLM:

```
http://localhost:8002/v1/chat/completions
```

### Benchmark Script

The benchmark logic is implemented in `~/z13-setup/benchmark.py` (stdlib only — no external dependencies). Batch scripts (`bench_new_batch.py`, `bench_batch3.py`) use `importlib.util.spec_from_file_location` to load the benchmark module without triggering its `__main__` block, then call `benchmark_model()` for each model. Results are saved to `~/benchmark_results.json` (the authoritative source of record for this report) and a summary table is printed.

---

## 9. Known Issues & Bug Reports

### Lemonade `/v1/rerank` not forwarded to backend llama-server

The Lemonade Server proxy does not forward `POST /api/v1/rerank` (or `/v1/rerank`) requests to the underlying llama-server process that hosts the reranker. To benchmark `bge-reranker-v2-m3`, the underlying llama-server port must be accessed directly (port 8003 in this deployment).

**Suggested fix:** Add `/v1/rerank` to the Lemonade proxy's route table, alongside `chat/completions` and `embeddings`.

### Lemonade vLLM proxy returns HTTP 500 on model load

The bundled vLLM backend (v0.20.1-rocm7.12.0) cannot be launched through the standard Lemonade proxy in this deployment — all model loads return HTTP 500. The companion vLLM reports use a direct-launch container/process as a workaround. See the [vLLM 512M report](Z13-vLLM-Benchmark-512M.md) and [vLLM 96G report](Z13-vLLM-Benchmark-96G.md) for full details, including a more specific bug (the proxy does not pass `--max-num-seqs`, which prevents Qwen3.5/Mamba-architecture models from initializing their cache blocks).

### Dense Q4_K_M >15B Vulkan kernel wall

Five distinct models (Devstral-22B, Qwen3.5-27B, Qwen3.6-27B, Qwen2.5-Coder-32B, Gemma-4-31B) all generate at 9–14 tok/s — roughly 5× slower than their parameter count would predict on this hardware. The wall persists across three Qwen generations and one Gemma generation, indicating a kernel-level limitation rather than a model-specific issue. MXFP4 and MoE variants in the same size class are unaffected (e.g., gpt-oss-20b-mxfp4 at 66 tok/s, Qwen3-30B-A3B at 76 tok/s).

**Suggested investigation areas:** matrix multiply kernel tile sizing for large `K` dimension in dense Q4_K_M; weight prefetch hints for Vulkan compute shaders on gfx1151.

---

## License & Reuse

This report is published under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Benchmark scripts and supporting code are MIT-licensed.

If you reuse these results in publications, presentations, or product decisions, attribution is appreciated:

> Pascua, R. (2026). *Z13 Local AI Stack — Lemonade Benchmark Report (Baseline: 512 MB UMA Buffer).*

This work is part of a community contribution to the [Lemonade by AMD](https://github.com/lemonade-sdk/lemonade) ecosystem and the AMD AI developer community. Bug reports and corrections are welcome.
