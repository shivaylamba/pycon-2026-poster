from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


NEON = {
    "cyan": "#22d3ee",
    "green": "#4ade80",
    "yellow": "#facc15",
    "orange": "#fb923c",
    "pink": "#f472b6",
    "purple": "#a78bfa",
    "blue": "#60a5fa",
    "red": "#f87171",
    "bg": "#08111f",
    "panel": "#0d1b2f",
    "grid": "#23344f",
    "text": "#e5f1ff",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate systems field guide visuals and poster.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    make_visuals(Path(args.results), Path(args.out) if args.out else Path(args.results) / "fieldguide_assets")


def make_visuals(results_dir: Path, out: Path) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(results_dir / "metrics.csv")
    if "status" in metrics.columns:
        metrics = metrics[metrics["status"].fillna("ok").eq("ok")].copy()
    metrics = add_energy_views(metrics)
    metrics.to_csv(results_dir / "metrics_enriched.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(results_dir / "summary_enriched.csv", index=False)
    paths = [
        plot_scaling(summary, out / "experiment_a_scaling_curves.png"),
        plot_frontier(summary, out / "energy_accuracy_frontier.png"),
        plot_quantization(summary, out / "experiment_b_quantization.png"),
        plot_quant_waterfall(summary, out / "experiment_b_efficiency_waterfall.png"),
        plot_heatmap(summary, out / "experiment_c_workload_heatmap.png"),
        plot_radar(summary, out / "experiment_c_radar.png"),
        plot_cpu_gpu(summary, out / "experiment_d_cpu_gpu.png"),
        plot_concurrency(results_dir, out / "experiment_d_concurrency.png"),
        plot_latency_waterfall(metrics, out / "experiment_d_latency_waterfall.png"),
        plot_power_timeline(results_dir, out / "experiment_e_power_timeline.png"),
        plot_latency_trace(summary, out / "experiment_e_latency_trace.png"),
        write_sankey(out / "experiment_e_energy_sankey.html"),
        write_sankey_svg(out / "experiment_e_energy_sankey.svg"),
        write_transformer_svg(out / "expanding_transformer.svg"),
        write_lifecycle_svg(out / "request_lifecycle.svg"),
        write_memory_svg(out / "memory_movement.svg"),
        build_poster(results_dir, out, out / "systems_tradeoffs_poster.html"),
    ]
    return paths


def add_energy_views(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cpu_rows = df[df["hardware_profile"].eq("cpu")]
    idle_w = float(cpu_rows["gpu_power_avg_w"].dropna().median()) if not cpu_rows.empty else 0.0
    elapsed = df["elapsed_s"].fillna(df["total_latency_s"]).fillna(0.0)
    df["gpu_idle_power_w_used"] = idle_w
    df["net_gpu_energy_j"] = (df["gpu_energy_j"].fillna(0.0) - idle_w * elapsed).clip(lower=0.0)
    df["active_device_energy_j"] = np.where(df["hardware_profile"].eq("cpu"), df["cpu_energy_j"].fillna(0.0), df["net_gpu_energy_j"])
    df["active_tokens_per_joule"] = np.where(df["active_device_energy_j"] > 0, df["output_tokens"].fillna(df["total_tokens"]) / df["active_device_energy_j"], 0.0)
    return df


def primary_gpu_profile(df: pd.DataFrame) -> str:
    profiles = [p for p in df.get("hardware_profile", pd.Series(dtype=str)).dropna().unique().tolist() if p != "cpu"]
    for preferred in ("l40s_gpu", "a100_gpu", "datacenter_gpu", "consumer_gpu"):
        if preferred in profiles:
            return preferred
    return profiles[0] if profiles else "a100_gpu"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group = ["model_id", "model_label", "family", "architecture", "params_b", "quantization", "precision_bits", "hardware_profile", "workload"]
    numeric = [
        "score", "is_correct", "total_latency_s", "tokens_per_sec", "tokens_per_joule", "active_tokens_per_joule",
        "active_device_energy_j", "net_gpu_energy_j", "total_energy_j", "gpu_power_avg_w", "gpu_mem_peak_mb",
        "cost_per_request_usd", "input_tokens", "output_tokens", "total_tokens",
    ]
    return df.groupby(group, dropna=False)[[c for c in numeric if c in df.columns]].mean().reset_index()


def dark_ax(ax):
    ax.set_facecolor(NEON["panel"])
    ax.figure.set_facecolor(NEON["bg"])
    ax.tick_params(colors=NEON["text"], labelsize=8)
    ax.xaxis.label.set_color(NEON["text"])
    ax.yaxis.label.set_color(NEON["text"])
    ax.title.set_color(NEON["text"])
    for spine in ax.spines.values():
        spine.set_color(NEON["grid"])
    ax.grid(color=NEON["grid"], alpha=0.55)


def save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=190, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def blank(path: Path, title: str, message: str = "Run the benchmark to populate this panel.") -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    dark_ax(ax)
    ax.text(0.5, 0.58, title, transform=ax.transAxes, ha="center", color=NEON["text"], fontsize=15, weight="bold")
    ax.text(0.5, 0.42, message, transform=ax.transAxes, ha="center", color="#8aa0bd", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return save(fig, path)


def plot_scaling(summary: pd.DataFrame, path: Path) -> Path:
    gpu = primary_gpu_profile(summary)
    data = summary[(summary.hardware_profile == gpu) & (summary.quantization == "q4") & (summary.workload == "code_generation")]
    if data.empty:
        return blank(path, "Experiment A: Scaling Curves")
    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    dark_ax(ax1)
    colors = [NEON["cyan"], NEON["green"], NEON["yellow"], NEON["pink"]]
    for (family, sub), color in zip(data.groupby("family"), colors):
        sub = sub.sort_values("params_b")
        ax1.plot(sub["params_b"], sub["score"], marker="o", color=color, label=f"{family} quality")
    ax2 = ax1.twinx()
    ax2.set_facecolor(NEON["panel"])
    ax2.tick_params(colors=NEON["orange"], labelsize=8)
    ax2.yaxis.label.set_color(NEON["orange"])
    for family, sub in data.groupby("family"):
        sub = sub.sort_values("params_b")
        ax2.plot(sub["params_b"], sub["active_device_energy_j"], marker="x", linestyle="--", color=NEON["orange"], alpha=0.45)
    ax1.set_xlabel("Parameters (billions)")
    ax1.set_ylabel("Code task score")
    ax2.set_ylabel("Net active energy/request (J)")
    ax1.set_title("Experiment A: Scaling Curves")
    ax1.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=NEON["text"])
    return save(fig, path)


def plot_frontier(summary: pd.DataFrame, path: Path) -> Path:
    gpu = primary_gpu_profile(summary)
    data = summary[(summary.hardware_profile == gpu) & (summary.workload == "code_generation")]
    if data.empty:
        return blank(path, "Energy vs Accuracy Frontier")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    dark_ax(ax)
    ax.scatter(data["active_device_energy_j"], data["score"], s=45 + data["params_b"] * 3, c=data["params_b"], cmap="viridis", alpha=0.9)
    for _, row in data.iterrows():
        ax.text(row["active_device_energy_j"], row["score"] + 0.015, row["model_label"].replace(" Q4", ""), fontsize=6, color=NEON["text"])
    ax.set_xlabel("Net active energy/request (J)")
    ax.set_ylabel("Quality score")
    ax.set_title("Energy vs Accuracy Frontier")
    return save(fig, path)


def plot_quantization(summary: pd.DataFrame, path: Path) -> Path:
    gpu = primary_gpu_profile(summary)
    data = summary[(summary.hardware_profile == gpu) & (summary.workload == "code_generation") & (summary.quantization.isin(["q4", "q8", "fp16"]))]
    data = data[data.family.isin(["Gemma", "Phi3", "CodeLlama"])]
    if data.empty:
        return blank(path, "Experiment B: Quantization Efficiency")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax in axes:
        dark_ax(ax)
    labels = data["model_label"].str.replace("CodeLlama", "CL", regex=False)
    colors = data["quantization"].map({"q4": NEON["green"], "q8": NEON["yellow"], "fp16": NEON["pink"]}).fillna(NEON["cyan"])
    axes[0].barh(labels, data["gpu_mem_peak_mb"] / 1024, color=colors)
    axes[0].set_title("VRAM footprint")
    axes[0].set_xlabel("Peak GB")
    axes[1].barh(labels, data["active_device_energy_j"], color=colors)
    axes[1].set_title("Energy/request")
    axes[1].set_xlabel("J")
    axes[2].barh(labels, data["active_tokens_per_joule"], color=colors)
    axes[2].set_title("Tokens/J")
    axes[2].set_xlabel("decoded tokens per J")
    fig.suptitle("Experiment B: Quantization Efficiency", color=NEON["text"], fontsize=13, weight="bold")
    return save(fig, path)


def plot_quant_waterfall(summary: pd.DataFrame, path: Path) -> Path:
    gpu = primary_gpu_profile(summary)
    data = summary[(summary.hardware_profile == gpu) & (summary.workload == "code_generation")]
    data = data[data.family.isin(["Gemma", "Phi3", "CodeLlama"]) & data.quantization.isin(["q4", "fp16"])]
    pairs = []
    for family, sub in data.groupby(["family", "params_b"], dropna=False):
        by_q = sub.set_index("quantization")
        if {"fp16", "q4"}.issubset(by_q.index):
            fp16 = by_q.loc["fp16"]
            q4 = by_q.loc["q4"]
            pairs.append({
                "label": f"{family[0]} {family[1]:g}B",
                "energy_drop": float(fp16["active_device_energy_j"] - q4["active_device_energy_j"]),
                "vram_drop": float((fp16["gpu_mem_peak_mb"] - q4["gpu_mem_peak_mb"]) / 1024),
                "score_delta": float(q4["score"] - fp16["score"]),
            })
    if not pairs:
        return blank(path, "Quantization Efficiency Waterfall", "Needs matching fp16 and q4 rows.")
    df = pd.DataFrame(pairs).sort_values("energy_drop")
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    dark_ax(ax)
    y = np.arange(len(df))
    ax.barh(y, df["energy_drop"], color=NEON["green"], label="energy saved/request (J)")
    ax.scatter(df["score_delta"] * max(df["energy_drop"].max(), 1), y, color=NEON["pink"], label="quality delta, scaled")
    ax.set_yticks(y, df["label"])
    ax.set_xlabel("Q4 savings versus FP16")
    ax.set_title("Quantization Efficiency Waterfall")
    ax.legend(frameon=False, fontsize=7, labelcolor=NEON["text"])
    return save(fig, path)


def plot_heatmap(summary: pd.DataFrame, path: Path) -> Path:
    gpu = primary_gpu_profile(summary)
    data = summary[(summary.hardware_profile == gpu) & (summary.params_b.between(6.0, 8.0))]
    pivot = data.pivot_table(index="model_label", columns="workload", values="score", aggfunc="mean").fillna(0)
    if pivot.empty:
        return blank(path, "Experiment C: Workload Specialization Matrix")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    dark_ax(ax)
    im = ax.imshow(pivot.values, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    ax.set_title("Experiment C: Workload Specialization Matrix")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    return save(fig, path)


def plot_radar(summary: pd.DataFrame, path: Path) -> Path:
    gpu = primary_gpu_profile(summary)
    data = summary[(summary.hardware_profile == gpu) & (summary.params_b.between(6.0, 8.0))]
    metrics = data.groupby("model_label").agg(score=("score", "mean"), speed=("tokens_per_sec", "mean"), efficiency=("active_tokens_per_joule", "mean")).fillna(0)
    if metrics.empty:
        return blank(path, "Architecture Radar")
    norm = metrics / metrics.max().replace(0, 1)
    labels = list(norm.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(5.8, 5.0), facecolor=NEON["bg"])
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(NEON["panel"])
    for idx, (model, row) in enumerate(norm.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, label=model, linewidth=1.8)
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1], labels, color=NEON["text"], fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["", "", "", ""], color=NEON["grid"])
    ax.grid(color=NEON["grid"])
    ax.set_title("Architecture Radar", color=NEON["text"], y=1.08)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6, frameon=False, labelcolor=NEON["text"])
    fig.savefig(path, dpi=190, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cpu_gpu(summary: pd.DataFrame, path: Path) -> Path:
    data = summary[summary.model_id.eq("codellama-7b-q4")]
    if data.empty:
        data = summary[summary.quantization.eq("q4")].head(12)
    pivot = data.pivot_table(index="workload", columns="hardware_profile", values="tokens_per_sec", aggfunc="mean").fillna(0)
    if pivot.empty:
        return blank(path, "Experiment D: CPU vs GPU Throughput")
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    dark_ax(ax)
    x = np.arange(len(pivot.index))
    width = 0.35
    gpu = primary_gpu_profile(summary)
    ax.bar(x - width / 2, pivot.get("cpu", pd.Series(0, index=pivot.index)), width, label="CPU", color=NEON["yellow"])
    ax.bar(x + width / 2, pivot.get(gpu, pd.Series(0, index=pivot.index)), width, label=gpu.replace("_", " ").upper(), color=NEON["cyan"])
    ax.set_xticks(x, pivot.index, rotation=20, ha="right")
    ax.set_ylabel("tokens/sec or items/sec")
    ax.set_title("Experiment D: CPU vs GPU Throughput")
    ax.legend(frameon=False, labelcolor=NEON["text"])
    return save(fig, path)


def plot_concurrency(results_dir: Path, path: Path) -> Path:
    csv_path = results_dir / "concurrency_metrics.csv"
    if not csv_path.exists():
        return blank(path, "Concurrency Scaling Curves", "Run scripts/fieldguide_concurrency.py to populate this panel.")
    data = pd.read_csv(csv_path)
    if data.empty:
        return blank(path, "Concurrency Scaling Curves")
    fig, ax1 = plt.subplots(figsize=(8.5, 3.6))
    dark_ax(ax1)
    for hw, sub in data.groupby("hardware_profile"):
        sub = sub.sort_values("concurrency")
        ax1.plot(sub["concurrency"], sub["tokens_per_sec"], marker="o", linewidth=2, label=f"{hw} tokens/s")
    ax1.set_xlabel("Concurrent requests")
    ax1.set_ylabel("Tokens/sec")
    ax2 = ax1.twinx()
    for hw, sub in data.groupby("hardware_profile"):
        sub = sub.sort_values("concurrency")
        ax2.plot(sub["concurrency"], sub["total_energy_j"], marker="x", linestyle="--", alpha=0.7, label=f"{hw} joules")
    ax2.tick_params(colors=NEON["yellow"], labelsize=8)
    ax2.yaxis.label.set_color(NEON["yellow"])
    ax2.set_ylabel("Total joules")
    ax1.set_title("Experiment D: Concurrency Scaling")
    ax1.legend(frameon=False, fontsize=7, labelcolor=NEON["text"], loc="upper left")
    return save(fig, path)


def plot_latency_waterfall(metrics: pd.DataFrame, path: Path) -> Path:
    data = metrics[metrics["model_id"].eq("codellama-7b-q4")]
    if data.empty:
        data = metrics[metrics["workload"].eq("chat_completion")]
    parts = ["load_s", "network_s", "tokenization_s", "inference_s", "postprocess_s"]
    keep = [col for col in parts if col in data.columns]
    if data.empty or not keep:
        return blank(path, "CPU vs GPU Latency Waterfall")
    agg = data.groupby(["hardware_profile", "workload"], dropna=False)[keep].mean().reset_index().head(8)
    labels = agg["hardware_profile"] + " · " + agg["workload"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    dark_ax(ax)
    left = np.zeros(len(agg))
    colors = [NEON["purple"], NEON["cyan"], NEON["yellow"], NEON["green"], NEON["pink"]]
    for col, color in zip(keep, colors):
        vals = agg[col].fillna(0).to_numpy()
        ax.barh(labels, vals, left=left, color=color, label=col.replace("_s", ""))
        left += vals
    ax.set_xlabel("Seconds")
    ax.set_title("CPU vs GPU Waterfall Latency Breakdown")
    ax.legend(frameon=False, fontsize=7, labelcolor=NEON["text"], ncol=3)
    return save(fig, path)


def plot_power_timeline(results_dir: Path, path: Path) -> Path:
    trace_path = results_dir / "traces" / "power_samples.jsonl"
    if not trace_path.exists():
        return blank(path, "GPU + CPU Power Utilization Timeline", "Run with --save-traces to populate this panel.")
    samples = pd.read_json(trace_path, lines=True)
    if samples.empty:
        return blank(path, "GPU + CPU Power Utilization Timeline")
    run_id = samples.groupby("run_id")["gpu_power_w"].max().sort_values().index[-1]
    data = samples[samples["run_id"].eq(run_id)].copy()
    fig, ax1 = plt.subplots(figsize=(8.5, 3.6))
    dark_ax(ax1)
    ax1.plot(data["t_rel_s"], data["gpu_power_w"], color=NEON["green"], linewidth=2, label="GPU watts")
    ax1.set_xlabel("Seconds from request start")
    ax1.set_ylabel("GPU watts")
    ax2 = ax1.twinx()
    ax2.plot(data["t_rel_s"], data["cpu_util_pct"], color=NEON["yellow"], linewidth=1.6, label="CPU util %")
    ax2.tick_params(colors=NEON["yellow"], labelsize=8)
    ax2.yaxis.label.set_color(NEON["yellow"])
    ax2.set_ylabel("CPU utilization %")
    ax1.set_title("Experiment E: GPU + CPU Power Timeline")
    return save(fig, path)


def plot_latency_trace(summary: pd.DataFrame, path: Path) -> Path:
    gpu = primary_gpu_profile(summary)
    data = summary[(summary.hardware_profile == gpu) & (summary.workload == "chat_completion")].sort_values("total_latency_s").tail(8)
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    dark_ax(ax)
    ax.barh(data["model_label"], data["total_latency_s"], color=NEON["blue"])
    ax.set_xlabel("Wall latency (s)")
    ax.set_title("Experiment E: Request Trace Timeline")
    return save(fig, path)


def write_sankey(path: Path) -> Path:
    labels = ["Prompt", "CPU tokenization", "RAM transfer", "VRAM load", "Transformer", "KV cache", "Sampling", "Response"]
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=14, thickness=14, color=["#22d3ee", "#facc15", "#60a5fa", "#a78bfa", "#4ade80", "#fb923c", "#f472b6", "#22d3ee"]),
        link=dict(source=[0, 1, 2, 3, 4, 4, 5, 6], target=[1, 2, 3, 4, 5, 6, 6, 7], value=[8, 6, 6, 18, 10, 22, 7, 5])
    )])
    fig.update_layout(title="Full System Energy Flow", paper_bgcolor=NEON["bg"], font_color=NEON["text"])
    fig.write_html(path)
    return path


def write_sankey_svg(path: Path) -> Path:
    stages = [
        ("Prompt", 7, NEON["cyan"]), ("CPU tokenization", 9, NEON["yellow"]),
        ("RAM transfer", 5, NEON["blue"]), ("VRAM load", 13, NEON["purple"]),
        ("Transformer", 38, NEON["green"]), ("KV cache", 14, NEON["orange"]),
        ("Sampling", 8, NEON["pink"]), ("Response", 6, NEON["cyan"]),
    ]
    x = 34
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 300"><rect width="1120" height="300" fill="{NEON["bg"]}"/>']
    parts.append(f'<text x="34" y="38" fill="{NEON["text"]}" font-size="25" font-weight="900">Sankey Energy Flow: One LLM Request</text>')
    last_mid = None
    for label, value, color in stages:
        h = 42 + value * 2.2
        y = 170 - h / 2
        w = 104 if label != "Transformer" else 142
        mid = (x, y + h / 2, w, color)
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{w}" height="{h:.1f}" rx="12" fill="#0d1b2f" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+w/2}" y="{y+h/2-4:.1f}" text-anchor="middle" fill="{NEON["text"]}" font-size="13" font-weight="800">{label}</text>')
        parts.append(f'<text x="{x+w/2}" y="{y+h/2+16:.1f}" text-anchor="middle" fill="#8aa0bd" font-size="12">{value}% signal</text>')
        if last_mid:
            lx, ly, lw, lc = last_mid
            sw = max(5, value / 2.6)
            parts.append(f'<path d="M {lx+lw+4} {ly:.1f} C {lx+lw+35} {ly:.1f}, {x-35} {y+h/2:.1f}, {x-4} {y+h/2:.1f}" fill="none" stroke="{color}" stroke-width="{sw:.1f}" opacity=".78"/>')
        last_mid = mid
        x += w + 34
    parts.append(f'<text x="34" y="268" fill="#8aa0bd" font-size="14">Stage widths are illustrative; benchmark CSVs provide measured latency, power, VRAM, and cost columns.</text></svg>')
    path.write_text("".join(parts), encoding="utf-8")
    return path


def write_transformer_svg(path: Path) -> Path:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 260">
<rect width="900" height="260" fill="{NEON['bg']}"/>
<text x="30" y="38" fill="{NEON['text']}" font-size="24" font-family="Inter,Arial" font-weight="800">Where Scaling Breaks</text>
<text x="30" y="66" fill="#8aa0bd" font-size="14">More parameters expand layers, KV cache, VRAM movement, and attention compute O(n^2).</text>
{''.join(f'<rect x="{40+i*95}" y="{100-i*4}" width="56" height="{70+i*9}" rx="8" fill="none" stroke="{NEON["cyan"]}" stroke-width="3"/><text x="{68+i*95}" y="210" fill="{NEON["text"]}" font-size="12" text-anchor="middle">{i+1}x</text>' for i in range(7))}
<path d="M690 88 C760 110 790 160 844 190" fill="none" stroke="{NEON['orange']}" stroke-width="5"/>
<text x="700" y="78" fill="{NEON['orange']}" font-size="18" font-weight="800">energy grows faster</text>
<text x="700" y="218" fill="{NEON['green']}" font-size="18" font-weight="800">quality gains flatten</text>
<text x="400" y="88" fill="{NEON['pink']}" font-size="36" font-weight="900">O(n²)</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")
    return path


def write_lifecycle_svg(path: Path) -> Path:
    stages = ["User Prompt", "CPU Tokenization", "RAM Transfer", "VRAM Loading", "Transformer Inference", "KV Cache", "Sampling", "Post Process", "Response"]
    x = 28
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 260"><rect width="1200" height="260" fill="{NEON["bg"]}"/>']
    parts.append(f'<text x="28" y="36" fill="{NEON["text"]}" font-size="24" font-weight="800">What Actually Happens During an LLM Request?</text>')
    for i, stage in enumerate(stages):
        width = 118 if i != 4 else 170
        color = [NEON["cyan"], NEON["yellow"], NEON["blue"], NEON["purple"], NEON["green"], NEON["orange"], NEON["pink"], NEON["blue"], NEON["cyan"]][i]
        parts.append(f'<rect x="{x}" y="86" width="{width}" height="74" rx="10" fill="#0d1b2f" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+width/2}" y="118" text-anchor="middle" fill="{NEON["text"]}" font-size="13" font-weight="800">{stage}</text>')
        parts.append(f'<text x="{x+width/2}" y="142" text-anchor="middle" fill="#8aa0bd" font-size="11">latency + joules</text>')
        if i < len(stages) - 1:
            parts.append(f'<path d="M{x+width+4} 123 H{x+width+32}" stroke="{color}" stroke-width="3" marker-end="url(#a)"/>')
        x += width + 34
    parts.append(f'<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{NEON["cyan"]}"/></marker></defs>')
    parts.append(f'<text x="28" y="218" fill="#8aa0bd" font-size="15">Observability view: each stage has hardware, latency, power, memory, and cost signals.</text></svg>')
    path.write_text("".join(parts), encoding="utf-8")
    return path


def write_memory_svg(path: Path) -> Path:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 300">
<rect width="900" height="300" fill="{NEON['bg']}"/>
<text x="30" y="42" fill="{NEON['text']}" font-size="25" font-weight="900">Memory Movement View</text>
<text x="30" y="70" fill="#8aa0bd" font-size="14">Quantization reduces weight traffic; long prompts expand KV cache; batching changes the economics.</text>
<rect x="42" y="115" width="170" height="90" rx="12" fill="#0d1b2f" stroke="{NEON['yellow']}" stroke-width="3"/><text x="127" y="150" text-anchor="middle" fill="{NEON['text']}" font-size="17" font-weight="800">CPU RAM</text><text x="127" y="176" text-anchor="middle" fill="#8aa0bd" font-size="13">tokens + prompts</text>
<rect x="365" y="92" width="190" height="136" rx="12" fill="#0d1b2f" stroke="{NEON['purple']}" stroke-width="3"/><text x="460" y="138" text-anchor="middle" fill="{NEON['text']}" font-size="18" font-weight="800">GPU VRAM</text><text x="460" y="166" text-anchor="middle" fill="#8aa0bd" font-size="13">weights + KV cache</text><text x="460" y="194" text-anchor="middle" fill="{NEON['green']}" font-size="13" font-weight="800">q4 &lt; q8 &lt; fp16</text>
<rect x="690" y="115" width="170" height="90" rx="12" fill="#0d1b2f" stroke="{NEON['green']}" stroke-width="3"/><text x="775" y="150" text-anchor="middle" fill="{NEON['text']}" font-size="17" font-weight="800">Tensor Cores</text><text x="775" y="176" text-anchor="middle" fill="#8aa0bd" font-size="13">attention + MLP</text>
<path d="M218 160 H356" stroke="{NEON['blue']}" stroke-width="8" opacity=".8"/><path d="M558 160 H682" stroke="{NEON['green']}" stroke-width="8" opacity=".8"/>
<text x="286" y="143" text-anchor="middle" fill="{NEON['blue']}" font-size="13" font-weight="800">prefill traffic</text>
<text x="620" y="143" text-anchor="middle" fill="{NEON['green']}" font-size="13" font-weight="800">decode loop</text>
<text x="365" y="260" fill="{NEON['pink']}" font-size="24" font-weight="900">KV cache grows with context length</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")
    return path


def build_poster(results_dir: Path, assets: Path, path: Path) -> Path:
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cost, Energy & Infrastructure Tradeoffs of Everyday LLM Workloads</title>
<style>
@page {{ size: 84.1cm 59.4cm landscape; margin: 0; }}
body {{ margin:0; background:#030712; font-family:Inter,ui-sans-serif,system-ui,sans-serif; color:#e5f1ff; }}
.poster {{ width:min(100vw,1580px); aspect-ratio:84.1/59.4; margin:auto; background:radial-gradient(circle at 20% 0%,#13213a,#07101f 48%,#030712); display:grid; grid-template-rows:86px 142px 1fr 138px; gap:10px; padding:18px; box-sizing:border-box; overflow:hidden; }}
.header {{ display:grid; grid-template-columns:1fr auto; align-items:start; border-bottom:2px solid #22d3ee; padding-bottom:10px; }}
h1 {{ margin:0; font-size:34px; letter-spacing:-.02em; line-height:1.05; }}
.subtitle {{ color:#8aa0bd; margin-top:7px; font-size:14px; }}
.thesis {{ color:#4ade80; font-weight:800; font-size:18px; text-align:right; max-width:460px; }}
.hero {{ display:grid; grid-template-columns:1fr 320px; gap:12px; }}
.hero img {{ width:100%; height:100%; object-fit:cover; border:1px solid #23344f; border-radius:10px; background:#0d1b2f; }}
.kpis {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.kpi {{ background:#0d1b2f; border:1px solid #23344f; border-radius:10px; padding:12px; }}
.kpi b {{ display:block; color:#22d3ee; font-size:24px; }} .kpi span {{ color:#8aa0bd; font-size:12px; }}
.grid {{ display:grid; grid-template-columns:1fr 1.25fr 1fr; gap:10px; min-height:0; }}
.col {{ display:flex; flex-direction:column; gap:10px; min-width:0; }}
.panel {{ background:rgba(13,27,47,.86); border:1px solid #23344f; border-radius:10px; padding:8px; min-height:0; overflow:hidden; }}
.panel h2 {{ margin:0 0 8px; font-size:16px; color:#e5f1ff; text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid #23344f; padding-bottom:6px; }}
.panel img {{ width:100%; max-height:230px; object-fit:contain; display:block; border-radius:6px; }}
.note {{ color:#8aa0bd; font-size:12px; line-height:1.35; }}
.callout {{ color:#facc15; font-size:15px; font-weight:800; line-height:1.25; }}
.bottom {{ display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:10px; }}
.insight {{ background:#0d1b2f; border:1px solid #23344f; border-radius:10px; padding:10px; font-size:13px; line-height:1.25; }}
.insight b {{ color:#4ade80; display:block; margin-bottom:4px; }}
</style></head><body><main class="poster">
<section class="header"><div><h1>Cost, Energy & Infrastructure Tradeoffs of Everyday LLM Workloads</h1><div class="subtitle">A Visual Python Field Guide for local LLM deployment, observability, and systems tradeoffs · PyCon US · Shivay Lamba / Suvrakamal Das</div></div><div class="thesis">The most accurate LLM is often not the most economically efficient one.</div></section>
<section class="hero"><img src="request_lifecycle.svg" alt="request lifecycle"><div class="kpis"><div class="kpi"><b>5</b><span>isolated dimensions: size, quantization, architecture, hardware, workload</span></div><div class="kpi"><b>tokens/J</b><span>practical deployment efficiency metric</span></div><div class="kpi"><b>NVML</b><span>GPU power, VRAM, utilization traces</span></div><div class="kpi"><b>RAPL</b><span>CPU power when available; labeled estimate otherwise</span></div></div></section>
<section class="grid">
<div class="col"><div class="panel"><h2>Scaling laws & diminishing returns</h2><img src="experiment_a_scaling_curves.png"></div><div class="panel"><h2>Energy vs accuracy frontier</h2><img src="energy_accuracy_frontier.png"><div class="callout">Scaling can push energy faster than quality.</div></div></div>
<div class="col"><div class="panel"><h2>Quantization efficiency</h2><img src="experiment_b_quantization.png"><div class="callout">4-bit quantized models often retain most quality at a fraction of memory and energy.</div></div><div class="panel"><h2>Quantization waterfall</h2><img src="experiment_b_efficiency_waterfall.png"></div><div class="panel"><h2>Architecture specialization</h2><img src="experiment_c_workload_heatmap.png"></div></div>
<div class="col"><div class="panel"><h2>CPU vs GPU deployment</h2><img src="experiment_d_cpu_gpu.png"></div><div class="panel"><h2>Latency waterfall</h2><img src="experiment_d_latency_waterfall.png"></div><div class="panel"><h2>System energy flow</h2><img src="experiment_e_energy_sankey.svg"></div></div>
</section>
<section class="bottom">
<div class="insight"><b>1. Diminishing returns</b>Larger models frequently add cost faster than accuracy.</div>
<div class="insight"><b>2. Quantization wins</b>Compression improves practical deployment efficiency.</div>
<div class="insight"><b>3. Workloads differ</b>Architecture choice depends on code, chat, summary, and search patterns.</div>
<div class="insight"><b>4. Systems problem</b>LLM deployment is infrastructure engineering, not just model selection.</div>
</section></main></body></html>"""
    path.write_text(html, encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
