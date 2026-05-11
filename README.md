# Cost & Energy of Everyday LLM Workloads

Notebook-first benchmark kit for the PyCon US poster:

**Cost & Energy of Everyday LLM Workloads: A Visual Field Guide for Python Developers**

This is intentionally separate from the replication package in the parent
directory. It reuses the research idea, not the paper's code.

## What It Measures

The benchmark runs everyday Python-developer LLM scenarios:

- chat completions
- document summarization
- short-label classification
- RAG-style query embedding and vector ranking
- batch embedding jobs

Each run writes a tidy `metrics.csv` with:

- latency buckets: `network_s`, `tokenization_s`, `inference_s`, `postprocess_s`
- token counts: `input_tokens`, `output_tokens`, `total_tokens`
- cost estimates: `total_cost_usd`, `cost_per_1k_tokens_usd`
- energy indicators: `cpu_energy_j`, `gpu_energy_j`, `net_gpu_energy_j`,
  `active_device_energy_j`, `total_energy_j`
- task fields such as classification correctness and retrieved RAG document ids

For Ollama, `tokenization_s` is the backend's prompt-evaluation/prefill time.
Ollama does not expose a pure tokenizer-only timer, so the raw field is kept
honest and poster labels use "Tokenization / prefill".

## Quick Smoke Test

The mock provider runs anywhere and verifies the full measurement and plotting
pipeline.

```bash
cd /Users/shivaylamba/Downloads/Replication_package/pycon_llm_cost_energy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_benchmark.py \
  --config configs/example_mock.yaml \
  --out results/mock_smoke \
  --limit 2 \
  --repeats 1

python scripts/make_plots.py \
  --metrics results/mock_smoke/metrics.csv \
  --out results/mock_smoke/figures
```

## Run On A CPU/GPU VM With Ollama

Install Ollama on the VM, then pull the example models or replace them in
`configs/example_ollama.yaml`.

```bash
ollama pull llama3.2:1b
ollama pull qwen2.5:3b
ollama pull mistral:7b-instruct-v0.3
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
```

Then run:

```bash
cd /Users/shivaylamba/Downloads/Replication_package/pycon_llm_cost_energy
source .venv/bin/activate

python scripts/run_benchmark.py \
  --config configs/example_ollama.yaml \
  --out results/ollama_vm \
  --repeats 3

python scripts/make_plots.py \
  --metrics results/ollama_vm/metrics.csv \
  --out results/ollama_vm/figures
```

The config defines two hardware profiles:

- `cpu`: passes `num_gpu: 0` to Ollama.
- `gpu`: lets Ollama use its automatic GPU placement and records NVIDIA power
  through NVML when available.

If Linux RAPL counters are readable, CPU package energy is measured directly.
If not, set `cpu_tdp_watts` in the hardware profile to get a labeled estimate.

The poster plots follow the energy-accounting style used by the referenced
LLM energy paper: sample device power over time, establish an idle baseline,
and report net work energy above idle. In generated figures,
`gpu_energy_j` remains the raw integrated NVML reading, while
`net_gpu_energy_j = gpu_energy_j - idle_gpu_watts * elapsed_s`. On the
Lightning A100 run, the idle GPU baseline is inferred from CPU-forced rows
because the A100 stays attached while Ollama runs with `num_gpu: 0`; CPU rows
use the explicit TDP/utilization estimate because RAPL counters were not
available in that VM.

## Notebooks

- `notebooks/llm_cost_energy_field_guide.ipynb`: run the full workflow from a
  notebook, starting with the mock provider and optionally switching to Ollama.
- `notebooks/poster_visuals_from_metrics.ipynb`: load any `metrics.csv`,
  regenerate figures, and produce summary tables for the poster.

## Adding Models

Edit `configs/example_ollama.yaml`:

```yaml
generation_models:
  - name: your-model-tag
    label: Your Model
    provider: ollama_local
    family: your-family
    size: 7B
```

For hosted or OpenAI-compatible APIs, add a provider with `type:
openai_compatible` and add pricing fields to each model:

```yaml
pricing:
  input_usd_per_1m_tokens: 0.15
  output_usd_per_1m_tokens: 0.60
```

Hosted APIs usually do not expose server-side tokenization/inference timings, so
their latency breakdowns are wall-time oriented. Local Ollama runs expose more
of the backend timing.

## Poster Heuristics This Supports

- Short classification calls are often prompt/network dominated.
- Long summarization calls are input-token dominated.
- Batch embeddings amortize request overhead.
- GPU watts alone are misleading: faster completion can reduce total joules.
- Cost per request and cost per thousand tokens answer different questions.
