# Lightning A100 mixed-family benchmark notes

Run target: NVIDIA A100-SXM4-40GB on Lightning.ai.

This run adds non-Qwen model families to the earlier Qwen-only sweep:

- Generation/chat workloads: llama3.2:3b, gemma3:4b, mistral-small3.2:24b
- Embedding/RAG workloads: nomic-embed-text:latest, mxbai-embed-large:latest, bge-m3:latest

CPU package energy counters were not available inside the Lightning VM, so CPU energy is estimated from the configured 120W CPU TDP and sampled process CPU utilization. GPU energy is measured with NVIDIA NVML. During CPU-profile runs, that GPU value reflects the attached A100's idle draw rather than inference work. Electricity cost uses $0.18/kWh.

Llama 4 Scout was not included because the official Docker Model Runner tag is about 67GB, which does not fit cleanly as a full-GPU comparison on this 40GB A100.
Active device energy columns were added after the run: for CPU rows this is the CPU TDP-based estimate, and for GPU rows this is NVML GPU energy. The original total_energy_j remains CPU estimate plus attached GPU NVML draw.
