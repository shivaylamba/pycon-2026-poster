# Archive Brief for PyCon US 2026 Poster

## Title
**Cost, Energy & Infrastructure Tradeoffs of Everyday LLM Workloads: A Visual Python Field Guide**

## Abstract
This repository is a notebook-first and script-first benchmark kit for Python developers who want to measure practical LLM tradeoffs: latency, cost, and energy. It reproduces a poster workflow and dashboard workflow using local Docker Model Runner-backed models, CPU/GPU hardware profiles, and workload-level instrumentation. Outputs include reproducible CSVs, charts, traces, and a printable poster/dashboard bundle.

## What this repository does
- Runs controlled benchmarks for everyday LLM tasks (chat, summarization, classification, RAG/semantic search, and embeddings).
- Splits latency into stages (`network_s`, `tokenization_s`, `inference_s`, `postprocess_s`).
- Computes token, cost, and energy views (`total_energy_j`, `net_gpu_energy_j`, `active_device_energy_j`, per-1k-token variants).
- Compares model families, quantization levels, hardware profiles, and system flow observability.
- Produces publication-ready visuals and an interactive field-guide dashboard.

## Research focus
The central research question is: **for Python-first, real-world LLM tasks, which model + hardware + quantization choices are fastest, cheapest, and most energy efficient—and when do these choices conflict?**

The field-guide experiments are organized as:
- **A: Model size scaling** within families
- **B: Quantization effects** (`fp16`, `q8`, `q4`)
- **C: Architecture comparison** at similar scales
- **D: Hardware comparison** (CPU vs GPU)
- **E: System flow observability** (request lifecycle and energy accounting)

## How we built and ran the benchmark
### Pipeline architecture
- Core benchmark package: `bench/`
- Field guide package: `fieldguide/`
- Entry scripts: `scripts/run_benchmark.py`, `scripts/fieldguide_run.py`, `scripts/fieldguide_make_visuals.py`, `scripts/fieldguide_concurrency.py`, `scripts/fieldguide_combine_results.py`
- Data/workloads: `data/`, `fieldguide/workloads.py`
- Exported runs and assets: `remote_results/`

### Benchmark method
1. Select config (`configs/*.yaml`) and hardware profiles.
2. Execute repeated workload items for each model/provider combination.
3. Capture latency buckets, tokens, cost estimates, and energy samples.
4. Aggregate to `metrics.csv` / `summary` tables.
5. Generate charts, poster HTML/PDF, and dashboard artifacts.

## NVIDIA GPU setup used in this repository
### A100 run track (Lightning.ai)
- Hardware: **NVIDIA A100-SXM4-40GB**
- Profiles: CPU-only and GPU deployment via Docker Model Runner provider configs
- Electricity pricing: `$0.18/kWh`
- GPU energy: NVML sampled
- CPU energy: TDP-based estimate when RAPL was unavailable
- Workloads: `chat`, `summarization`, `classification`, `rag_rank`, `batch_embeddings`

### Field-guide dashboard track
- Hardware export (`remote_results/fieldguide_l40s_combined/hardware.json`):
  - **NVIDIA L40S (44.99 GB)**
  - 16 logical CPU cores, 124 GB RAM
- Includes concurrency traces and a full systems dashboard export.

## Main benchmark types we looked at
Across repo tracks, the benchmark families include:
- **Generation tasks**: chat completion, code generation, summarization
- **Retrieval-style tasks**: semantic search / RAG rank
- **Embedding tasks**: embedding generation and embedding search
- **Infrastructure dimensions**: latency breakdown, energy accounting, cost per request, cost per 1k tokens, tokens per joule, active-device energy, CPU vs GPU behavior, and concurrency scaling

## LLMs used
### A100 combined run (Qwen + mixed families)
- Generation/chat: `qwen3.5:0.8b`, `qwen3.5:4b`, `qwen3.5:9b`, `llama3.2:3b`, `gemma3:4b`, `mistral-small3.2:24b`
- Embeddings: `qwen3-embedding:0.6b`, `qwen3-embedding:4b`, `nomic-embed-text:latest`, `mxbai-embed-large:latest`, `bge-m3:latest`

### Field-guide model registry (poster track)
Includes 19 model specs across families and quantization levels, including:
- Gemma, Phi-3, Granite Code, CodeLlama, Mistral, DeepSeek Coder
- Embedding models: Nomic Embed Text, mxbai Embed Large
- Quantization variants (`q4`, `q8`, `fp16`) for targeted comparison experiments

## Major takeaways from charts and matrices
### 1) GPU acceleration is often dominant for generation workloads
In the A100 combined summary, CPU/GPU latency speedups average ~4.9x, with peaks above 11x for larger models (for example Qwen 4B/9B summarization and classification).

### 2) Speedup is workload-dependent, not universal
Some embedding/RAG cases show limited or negative GPU benefit (for example Nomic Embed Text batch embeddings/RAG in this run), showing that small/bound workloads may remain overhead dominated.

### 3) Size strongly increases latency and energy
Within a family (A100 track), moving from smaller to larger Qwen variants materially increases latency and active-device joules.

### 4) Quantization can outperform fp16 on practical efficiency
Field-guide quantization tables show repeated cases where `q4`/`q8` variants are faster than `fp16` for chat, code generation, and summarization, while reducing active energy per task.

### 5) “Raw GPU joules” can be misleading without baseline handling
This repository explicitly tracks:
- `gpu_energy_j` (raw sampled)
- `net_gpu_energy_j` (idle-adjusted)
- `active_device_energy_j` (primary device work energy)
This separation is essential because attached idle GPU draw can distort CPU-profile comparisons.

### 6) There is no single “best model” across all criteria
Fastest, cheapest, and most accurate choices differ by workload. The dashboard/matrix outputs are designed to help pick models by scenario, not by one global rank.

## Expected audience Q&A (presentation-ready)
### Q1) What is new here versus generic LLM benchmarks?
This benchmark is Python-developer oriented, task-realistic, and systems-aware: it combines latency decomposition, energy accounting, and cost into one reproducible workflow and visual field guide.

### Q2) Why measure both `gpu_energy_j` and `net_gpu_energy_j`?
Raw GPU joules include baseline idle draw from an attached device. Net GPU energy removes estimated idle to better represent inference work.

### Q3) Why can GPU be slower for some embedding/RAG cases?
For lightweight requests, launch/runtime overhead can dominate. In those cases, CPU can be competitive or faster unless concurrency and batching are high.

### Q4) What did quantization show?
Across the field-guide experiment matrix, quantized variants (especially `q4`, often `q8`) frequently delivered better latency/energy tradeoffs than fp16 for practical workloads.

### Q5) Which workloads benefit most from GPU in your data?
Longer generation workloads (chat/summarization/code generation), especially with larger models, showed the largest gains.

### Q6) How should teams pick a model from your charts?
Start with workload type and SLA, then compare latency + active energy + cost together. Avoid single-metric decisions.

### Q7) Did you benchmark only one GPU?
No. The repository includes A100-focused runs and a full L40S field-guide export; the framework supports CPU and multiple GPU profiles.

### Q8) Can others reproduce this quickly?
Yes. The repo includes runnable configs, scripts, notebooks, and exported `remote_results/` artifacts so teams can rerun or extend the workflow.

### Q9) What are the top practical outcomes for engineering teams?
- Use GPU intentionally for heavy generation workloads.
- Validate overhead-sensitive tasks before assuming GPU wins.
- Use quantization as a first-class optimization lever.
- Track net/active energy, not only wall power.
- Pick models per workload objective (latency, cost, or quality), not by parameter count alone.
