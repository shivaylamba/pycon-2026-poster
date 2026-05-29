# Lightning A100 Benchmark Notes

Run target: NVIDIA A100-SXM4-40GB on Lightning AI.

Models used:

- `qwen3.5:0.8b`
- `qwen3.5:4b`
- `qwen3.5:9b`
- `qwen3-embedding:0.6b`
- `qwen3-embedding:4b`

`llama4:scout` was not run because Docker Model Runner lists it at 67 GB, which exceeds the 40 GB A100 VRAM in this session for a clean full-GPU comparison.

Energy notes:

- GPU energy is sampled from NVIDIA NVML.
- Linux RAPL CPU package energy was unavailable in this virtualized Lightning environment.
- CPU energy in `metrics.csv`, `benchmark_summary.csv`, and regenerated figures is a labeled estimate: `elapsed_s * 120W * cpu_util_avg_pct / 100`.
- The original unestimated metrics are preserved in `metrics_raw_before_cpu_estimate.csv`.

Run size: 248 metric rows across CPU and A100 GPU profiles, two repeats, five workloads.
Active device energy columns were added after the run: for CPU rows this is the CPU TDP-based estimate, and for GPU rows this is NVML GPU energy. The original total_energy_j remains CPU estimate plus attached GPU NVML draw.
