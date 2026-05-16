# Z13 Local AI Stack — Benchmark Series

A community deep-dive into running Lemonade Server, llama.cpp/Vulkan, and vLLM on the **ASUS ROG Flow Z13 GZ302EA** — AMD Ryzen AI MAX+ 395 (Strix Halo) with the Radeon 8060S iGPU (gfx1151, RDNA 3.5) and XDNA 2 NPU.

The goal: give every Strix Halo developer a reproducible reference for choosing models, inference backends, and BIOS/kernel configurations on this hardware.

## The Reports

| # | Report | What it covers |
|---|---|---|
| 1 | [**Lemonade Benchmark · 512 MB UMA**](Z13-Lemonade-Benchmark-512M.md) | Baseline run: 38 generative models + 2 utility models on Lemonade v10.4.0 with the default 512 MB UMA buffer (dynamic GTT). Establishes the throughput, prefill, and TTFT reference numbers, identifies the dense Q4_K_M >15B Vulkan kernel wall, and includes architectural decision rules. |
| 2 | [**Lemonade Benchmark · 96 GB Dedicated VRAM**](Z13-Lemonade-Benchmark-96G.md) | Same 38 models re-tested after switching to 96 GB dedicated VRAM + `transparent_hugepage=always` + `numa_balancing=disabled` + tuned TTM page limits. Every model improves; top gain +22.5%. |
| 3 | [**vLLM Benchmark · 512 MB UMA**](Z13-vLLM-Benchmark-512M.md) | 8 models on vLLM (TheRock gfx1151 container, v0.19.2rc1 / ROCm 7.12.0) under the baseline configuration. Includes a full vLLM-vs-GGUF analysis and explains why single-stream vLLM is 51–85% slower than llama.cpp on this hardware. |
| 4 | [**vLLM Benchmark · 96 GB Dedicated VRAM**](Z13-vLLM-Benchmark-96G.md) | 2 Qwen3.5 hybrid Mamba+Attention models on the vLLM build bundled inside the Lemonade snap, run under 96 GB dedicated VRAM with a Mamba-cache workaround. Documents a Lemonade vLLM proxy bug. |

## Headline Findings

1. **The fastest model on this platform is LFM2-1.2B** (Liquid AI's SSM MoE) at 224.7 tok/s under the tuned configuration — 17% faster than the closest Transformer competitor (Qwen3.5-0.8B at 215.2 tok/s).

2. **Dense Q4_K_M models above ~15B hit a hard wall on the Vulkan backend.** Five distinct models (Qwen3.5-27B, Qwen3.6-27B, Qwen2.5-Coder-32B, Devstral-22B, Gemma-4-31B) all generate at 9–14 tok/s, while MoE/MXFP4 variants at the same size run at 50–80 tok/s. This wall holds across three Qwen generations plus Gemma — it is kernel-level, not model-specific.

3. **Switching to 96 GB dedicated VRAM with kernel tuning improves every model.** Average +10% generation throughput, +20% prefill throughput on small models, +30–38% prefill on large dense models, and 15–28% lower TTFT on the slowest-prefilling models.

4. **vLLM is for concurrent serving, not local single-user chat.** Single-stream vLLM throughput is 51–85% lower than llama.cpp on this hardware. Use Lemonade/llama.cpp for interactive use; reach for vLLM only when serving multiple concurrent users, agentic pipelines, or structured-output generation.

5. **Two reproducible Lemonade bugs were uncovered and documented:**
   - The vLLM proxy does not pass `--max-num-seqs` to vllm-server, blocking all Qwen3.5/Mamba models.
   - The Lemonade proxy does not forward `POST /v1/rerank` to the underlying llama-server (must access backend port directly).

## How to Use This Series

- **Picking a model for a new project?** Start with the [512 MB report's specialist rankings](Z13-Lemonade-Benchmark-512M.md#specialist-rankings) and architectural decision rules.
- **Tuning a Strix Halo system?** Apply the [96 GB report's BIOS + kernel recipe](Z13-Lemonade-Benchmark-96G.md#1-configuration-change-summary). All 38 models improved; none regressed.
- **Deciding llama.cpp vs vLLM?** Read the [vLLM-vs-GGUF analysis in the 512 MB vLLM report](Z13-vLLM-Benchmark-512M.md#3-vllm-vs-gguf-analysis).
- **Building with Qwen3.5/hybrid Mamba models on vLLM?** The [96 GB vLLM report](Z13-vLLM-Benchmark-96G.md) documents the Mamba cache block issue and a working workaround.

## Reproducibility

All four reports share:
- A standardized prompt (with `/no_thinking` appended for Qwen3.x models)
- 1 warm-up + 3 measured runs, median reported
- `max_tokens=256`, `temperature=0.0` (greedy decoding)
- Server-side timings where available (Lemonade); client-side wall-clock when not (vLLM)
- Sequential runs, no parallelism

Benchmark scripts (stdlib-only Python, no external dependencies) are available alongside this repository.

## License

Reports are published under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Benchmark scripts are MIT-licensed. Attribution appreciated:

> Pascua, R. (2026). *Z13 Local AI Stack — Benchmark Series.*

## Acknowledgements

- The [Lemonade team at AMD](https://github.com/lemonade-sdk/lemonade) for the inference platform.
- [@kyuz0](https://github.com/kyuz0) for the `vllm-therock-gfx1151` container image that made vLLM benchmarking possible on this hardware.
- The Liquid AI team for the LFM2 SSM MoE architecture, which surprises everyone who benchmarks it on iGPU.
- The AMD Developer Program and the [Lemonade Developer Challenge](https://www.amd.com/en/developer/resources/technical-articles/2026/join-the-lemonade-developer-challenge.html) for motivating this write-up.

Bug reports, corrections, and additional model requests are welcome.
