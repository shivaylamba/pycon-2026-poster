# Cost & Energy of Everyday LLM Workloads

Notebook-first benchmark kit for the PyCon US poster:

**Cost & Energy of Everyday LLM Workloads: A Visual Field Guide for Python Developers**

This is intentionally separate from the replication package in the parent
directory. It reuses the research idea, not the paper's code.

## Live Dashboard

Explore the full companion UI here:

**https://fieldguidel40scombined.vercel.app**

The dashboard expands the poster into a multi-page systems field guide with
latency waterfalls, measured GPU-energy charts, quantization comparisons,
CPU/GPU deployment views, request traces, downloadable CSVs, and a searchable
benchmark table.

## Energy Attribution Correction

The completed Lightning.ai L40S export in this repository does **not** contain
measured CPU package energy. The VM did not expose Linux RAPL counters, so CPU
rows are marked `cpu:tdp_estimate`: a coarse `elapsed × configured TDP × CPU
utilization` estimate. Those estimates remain in the CSV for transparency, but
they are excluded from measured-energy poster/dashboard claims.

What is measured in the current export:

- GPU power, utilization, and VRAM via NVIDIA NVML.
- Sampling interval: `0.1s`, approximately `10Hz`.
- Idle-adjusted GPU energy: `net_gpu_energy_j = gpu_energy_j - idle_gpu_watts × elapsed_s`.

What is not measured in the current export:

- CPU package joules, because RAPL was unavailable inside the VM.
- Full wall-plug system energy. Use a wall meter or cloud/provider power data
  for that.

If you rerun on bare metal with readable
`/sys/class/powercap/intel-rapl:*` counters, the CSV will mark CPU energy as
`cpu:rapl`. Until then, CPU-vs-GPU sections should be read as latency,
throughput, and deployment guidance, not measured CPU-energy evidence.

The generated static HTML also lives in this repository at
`remote_results/fieldguide_l40s_combined/fieldguide_assets/fieldguide_dashboard.html`.

## Systems Field Guide Track

The newer poster track is:

**Cost, Energy & Infrastructure Tradeoffs of Everyday LLM Workloads: A Visual Python Field Guide**

It benchmarks five independent deployment dimensions:

- **A: model size** within the same family: Gemma, Phi-3, Granite Code, CodeLlama
- **B: quantization** for the same model: fp16 vs q8 vs q4
- **C: architecture** at similar scale: Gemma, Mistral, CodeLlama, DeepSeek Coder
- **D: hardware**: CPU-only vs datacenter GPU (`l40s_gpu` / `a100_gpu`), with consumer GPU profiles supported by the same code
- **E: system flow**: request lifecycle, power traces, memory movement, and energy accounting

Run it on a Lightning.ai GPU VM:

```bash
cd /home/zeus/pycon_llm_cost_energy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/fieldguide_pull_models.py --experiment all --include-embeddings

python scripts/fieldguide_run.py \
  --experiment all \
  --hardware l40s_gpu \
  --out results/fieldguide_l40s_full \
  --limit 3 \
  --repeats 1 \
  --skip-pull \
  --save-traces

python scripts/fieldguide_run.py \
  --experiment D \
  --hardware l40s_gpu,cpu \
  --models codellama-7b-q4 \
  --out results/fieldguide_l40s_cpu_gpu_d \
  --limit 3 \
  --repeats 1 \
  --skip-pull \
  --save-traces

python scripts/fieldguide_concurrency.py \
  --model codellama-7b-q4 \
  --hardware l40s_gpu,cpu \
  --out results/fieldguide_l40s_cpu_gpu_d \
  --levels 1,2,4
```

The final visualization pass expects a `metrics.csv` plus optional
`concurrency_metrics.csv` and `traces/power_samples.jsonl`:

```bash
python scripts/fieldguide_combine_results.py \
  --inputs results/fieldguide_l40s_full,results/fieldguide_l40s_cpu_gpu_d \
  --out results/fieldguide_l40s_combined

python scripts/fieldguide_make_visuals.py \
  --results results/fieldguide_l40s_combined \
  --out results/fieldguide_l40s_combined/fieldguide_assets
```

It writes the dark observability-style printable poster to
`fieldguide_assets/systems_tradeoffs_poster.html`.

This repository also includes a completed L40S benchmark export under
`remote_results/fieldguide_l40s_combined/`, including reproducible CSVs,
power traces, PNG/SVG assets, and a screenshot preview of the final poster.

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
- energy indicators: `gpu_energy_j`, `net_gpu_energy_j`,
  `active_device_energy_j` for measured GPU rows, plus
  `cpu_energy_source`, `cpu_energy_estimate_j`, and `cpu_energy_measured_j`
  so CPU estimates are not confused with measured RAPL values
- task fields such as classification correctness and retrieved RAG document ids

Docker Model Runner uses an OpenAI-compatible API (via the standard `openai`
Python SDK). Server-side timing buckets such as tokenization and decode are not
exposed, so all request latency is attributed to wall-clock time.

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

## Run On A CPU/GPU VM With Docker Model Runner

Install Docker Model Runner on the VM, then pull the example models or replace them in
`configs/example_docker_model_runner.yaml`.

```bash
docker model pull llama3.2:1b
docker model pull qwen2.5:3b
docker model pull mistral:7b-instruct-v0.3
docker model pull nomic-embed-text
docker model pull mxbai-embed-large
```

Then run:

```bash
cd /Users/shivaylamba/Downloads/Replication_package/pycon_llm_cost_energy
source .venv/bin/activate

python scripts/run_benchmark.py \
  --config configs/example_docker_model_runner.yaml \
  --out results/docker_model_runner_vm \
  --repeats 3

python scripts/make_plots.py \
  --metrics results/docker_model_runner_vm/metrics.csv \
  --out results/docker_model_runner_vm/figures
```

The config defines two hardware profiles:

- `cpu`: CPU-only deployment profile.
- `gpu`: lets Docker Model Runner use its automatic GPU placement and records NVIDIA power
  through NVML when available.

If Linux RAPL counters are readable, CPU package energy is measured directly.
If not, `cpu_tdp_watts` only creates a labeled estimate. Do not use that
estimate as measured CPU energy, especially inside a VM.

The poster plots follow the energy-accounting style used by the referenced
LLM energy paper: sample device power over time, establish an idle baseline,
and report net work energy above idle. In generated figures,
`gpu_energy_j` remains the raw integrated NVML reading, while
`net_gpu_energy_j = gpu_energy_j - idle_gpu_watts * elapsed_s`. On Lightning
VMs, the idle GPU baseline can be inferred from CPU-forced rows because the GPU
stays attached while Docker Model Runner runs CPU-only. CPU rows in the bundled
exports use the explicit TDP/utilization estimate because RAPL counters were
not available; they are not plotted as measured energy in the corrected poster
assets.

Model-comparison summaries also exclude cold Docker Model Runner load rows (`load_s > 2s`)
so first-request model loading does not distort steady-state latency or energy
claims. The raw `metrics_enriched.csv` keeps those rows and flags them with
`is_cold_load_row`.

## Notebooks

- `notebooks/llm_cost_energy_field_guide.ipynb`: run the full workflow from a
  notebook, starting with the mock provider and optionally switching to Docker Model Runner.
- `notebooks/poster_visuals_from_metrics.ipynb`: load any `metrics.csv`,
  regenerate figures, and produce summary tables for the poster.

## Adding Models

Edit `configs/example_docker_model_runner.yaml`:

```yaml
generation_models:
  - name: your-model-tag
    label: Your Model
    provider: docker_model_runner_local
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
their latency breakdowns are wall-time oriented. Docker Model Runner also uses an
OpenAI-compatible API, so its latency breakdowns are wall-time oriented as well.

## Poster Heuristics This Supports

- Short classification calls are often prompt/network dominated.
- Long summarization calls are input-token dominated.
- Batch embeddings amortize request overhead.
- GPU watts alone are misleading: faster completion can reduce total joules.
- CPU energy on VMs needs RAPL or wall-meter validation; TDP/utilization
  estimates are only rough labels.
- Cost per request and cost per thousand tokens answer different questions.
