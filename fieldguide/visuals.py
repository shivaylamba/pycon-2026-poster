from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


NEON = {
    "cyan": "#0e7490",
    "green": "#15803d",
    "yellow": "#ca8a04",
    "orange": "#ea580c",
    "pink": "#be185d",
    "purple": "#7e22ce",
    "blue": "#1d4ed8",
    "red": "#b91c1c",
    "bg": "#ffffff",
    "panel": "#ffffff",
    "grid": "#d8dee8",
    "text": "#10213d",
}

POSTER_COLORS = {
    "navy": "#1a2744",
    "ink": "#10213d",
    "body": "#2c3e50",
    "muted": "#6c7a89",
    "border": "#d1d5db",
    "surface": "#f8f9fa",
    "blue": "#2563eb",
    "green": "#16a34a",
    "orange": "#f97316",
    "purple": "#7c3aed",
    "red": "#dc2626",
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
        plot_learning1_scaling_tradeoffs(summary, out / "learning1_scaling_tradeoffs.png"),
        plot_learning2_quantization_tradeoffs(summary, out / "learning2_quantization_tradeoffs.png"),
        plot_learning3_workload_specialization(summary, out / "learning3_workload_specialization.png"),
        plot_workload_active_energy(summary, out / "workload_active_device_energy.png"),
        plot_learning3_cpu_gpu_tradeoff(summary, out / "learning3_cpu_gpu_tradeoff.png"),
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
        write_lifecycle_svg(metrics, out / "request_lifecycle.svg"),
        write_memory_svg(out / "memory_movement.svg"),
        build_story_poster(results_dir, out, out / "systems_tradeoffs_poster.html"),
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
    token_basis = np.where(df["output_tokens"].fillna(0) > 0, df["output_tokens"].fillna(0), df["total_tokens"].fillna(0))
    df["active_tokens_per_joule"] = np.where(df["active_device_energy_j"] > 0, token_basis / df["active_device_energy_j"], 0.0)
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
    light_ax(ax)


def light_ax(ax):
    ax.set_facecolor(POSTER_COLORS["surface"])
    ax.figure.set_facecolor("#ffffff")
    ax.tick_params(colors=POSTER_COLORS["body"], labelsize=8)
    ax.xaxis.label.set_color(POSTER_COLORS["body"])
    ax.yaxis.label.set_color(POSTER_COLORS["body"])
    ax.title.set_color(POSTER_COLORS["ink"])
    for spine in ax.spines.values():
        spine.set_color(POSTER_COLORS["border"])
    ax.grid(color=POSTER_COLORS["border"], alpha=0.65, linewidth=0.7)


def save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=190, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def blank(path: Path, title: str, message: str = "Run the benchmark to populate this panel.") -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    light_ax(ax)
    ax.text(0.5, 0.58, title, transform=ax.transAxes, ha="center", color=POSTER_COLORS["ink"], fontsize=15, weight="bold")
    ax.text(0.5, 0.42, message, transform=ax.transAxes, ha="center", color=POSTER_COLORS["muted"], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return save(fig, path)


def workload_label(value: str) -> str:
    return {
        "chat_completion": "Chat",
        "code_generation": "Code",
        "summarization": "Summary",
        "semantic_search": "RAG",
        "embedding_search": "Embeddings",
    }.get(value, value.replace("_", " ").title())


def quant_color(value: str) -> str:
    return {"q4": POSTER_COLORS["green"], "q8": POSTER_COLORS["orange"], "fp16": POSTER_COLORS["navy"]}.get(value, POSTER_COLORS["blue"])


def plot_learning1_scaling_tradeoffs(summary: pd.DataFrame, path: Path) -> Path:
    gpu = primary_gpu_profile(summary)
    data = summary[(summary.hardware_profile == gpu) & (summary.quantization == "q4") & (summary.workload == "code_generation")].copy()
    families = ["Gemma", "Phi3", "Granite", "Granite Code", "CodeLlama"]
    data = data[data.family.isin(families)]
    if data.empty:
        return blank(path, "Learning 1: Bigger Models Have Diminishing Returns")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.9), sharex=True)
    colors = {
        "Gemma": POSTER_COLORS["blue"],
        "Phi3": POSTER_COLORS["green"],
        "Granite": POSTER_COLORS["orange"],
        "Granite Code": POSTER_COLORS["orange"],
        "CodeLlama": POSTER_COLORS["purple"],
    }
    panels = [
        ("score", "Toy/code score", "higher is better", None),
        ("active_device_energy_j", "Energy/request (J)", "lower is better", None),
        ("total_latency_s", "Latency/request (s)", "lower is better", None),
        ("gpu_mem_peak_mb", "Peak VRAM (GB)", "lower is better", 1024.0),
    ]
    for ax, (metric, title, note, divisor) in zip(axes.flat, panels):
        light_ax(ax)
        for family in families:
            sub = data[data.family.eq(family)].sort_values("params_b")
            if sub.empty:
                continue
            y = sub[metric] / divisor if divisor else sub[metric]
            ax.plot(sub["params_b"], y, marker="o", linewidth=2.2, markersize=5.2, color=colors.get(family, POSTER_COLORS["blue"]), label=family)
        ax.set_title(f"{title} ({note})", fontsize=10, weight="bold")
        ax.set_xlabel("Parameters (billions)")
        ax.set_ylabel(title)
        if metric == "score":
            ax.set_ylim(-0.03, 1.03)
    axes[0, 0].legend(loc="best", fontsize=7, frameon=True, facecolor="#ffffff", edgecolor=POSTER_COLORS["border"])
    fig.suptitle("Learning 1: same-family Q4 scaling on the code-generation toy set", fontsize=14, weight="bold", color=POSTER_COLORS["ink"])
    fig.text(0.5, 0.018, "Each panel uses the current primary-GPU, q4, code-generation filter. Quality is a toy pass rate; energy, latency, and VRAM are measured per request.", ha="center", fontsize=8.6, color=POSTER_COLORS["body"])
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(path, dpi=190, facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_learning2_quantization_tradeoffs(summary: pd.DataFrame, path: Path) -> Path:
    gpu = primary_gpu_profile(summary)
    data = summary[(summary.hardware_profile == gpu) & (summary.quantization.isin(["q4", "q8", "fp16"]))].copy()
    wanted = (
        ((data.family == "Gemma") & (data.params_b == 7.0)) |
        ((data.family == "Phi3") & (data.params_b == 14.0)) |
        ((data.family == "CodeLlama") & (data.params_b == 7.0))
    )
    data = data[wanted]
    if data.empty:
        return blank(path, "Learning 2: Quantization Changes the Economics")

    same_model_keys = ["family", "architecture", "params_b"]
    agg = data.groupby(same_model_keys + ["quantization"], dropna=False).agg(
        score=("score", "mean"),
        energy=("active_device_energy_j", "mean"),
        latency=("total_latency_s", "mean"),
        throughput=("tokens_per_sec", "mean"),
        vram=("gpu_mem_peak_mb", "mean"),
    ).reset_index()
    required_quants = {"fp16", "q8", "q4"}
    complete = []
    for _, sub in agg.groupby(same_model_keys, dropna=False):
        if required_quants.issubset(set(sub["quantization"])):
            complete.append(sub)
    if not complete:
        return blank(path, "Learning 2: Quantization Changes the Economics", "Needs matching fp16, q8, and q4 rows for the same models.")
    agg = pd.concat(complete, ignore_index=True)
    rows = []
    for _, sub in agg.groupby(same_model_keys, dropna=False):
        by_q = sub.set_index("quantization")
        baseline = by_q.loc["fp16"]
        baseline_score = max(float(baseline["score"]), 1e-9)
        for quantization in ["fp16", "q8", "q4"]:
            row = by_q.loc[quantization]
            rows.append({
                "quantization": quantization,
                "VRAM": 100 * row["vram"] / max(float(baseline["vram"]), 1e-9),
                "Energy": 100 * row["energy"] / max(float(baseline["energy"]), 1e-9),
                "Latency": 100 * row["latency"] / max(float(baseline["latency"]), 1e-9),
                "Throughput": 100 * row["throughput"] / max(float(baseline["throughput"]), 1e-9),
                "Quality": 100 * row["score"] / baseline_score,
            })
    normalized = pd.DataFrame(rows)
    if normalized.empty:
        return blank(path, "Learning 2: Quantization Changes the Economics")
    plot_df = normalized.groupby("quantization")[["VRAM", "Energy", "Latency", "Throughput", "Quality"]].mean().reindex(["fp16", "q8", "q4"]).dropna(how="all")

    fig, (ax_cost, ax_upside) = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.28, 1.0]})
    for ax in (ax_cost, ax_upside):
        light_ax(ax)

    cost_metrics = ["VRAM", "Energy", "Latency"]
    x = np.arange(len(cost_metrics))
    width = 0.24
    for offset, quantization in zip([-width, 0, width], ["fp16", "q8", "q4"]):
        if quantization not in plot_df.index:
            continue
        vals = plot_df.loc[quantization, cost_metrics].to_numpy()
        bars = ax_cost.bar(x + offset, vals, width=width, color=quant_color(quantization), label=quantization.upper())
        for bar, value in zip(bars, vals):
            ax_cost.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{value:.0f}%", ha="center", va="bottom", fontsize=7, color=POSTER_COLORS["ink"], weight="bold")
    ax_cost.axhline(100, color=POSTER_COLORS["border"], linewidth=1.2, linestyle="--")
    ax_cost.set_xticks(x, ["VRAM", "Energy", "Latency"])
    ax_cost.set_ylim(0, 118)
    ax_cost.set_ylabel("% of the same model in FP16")
    ax_cost.set_title("Costs to minimize: lower bars are better", fontsize=11, weight="bold")

    upside_metrics = ["Throughput", "Quality"]
    x2 = np.arange(len(upside_metrics))
    for offset, quantization in zip([-width, 0, width], ["fp16", "q8", "q4"]):
        if quantization not in plot_df.index:
            continue
        vals = plot_df.loc[quantization, upside_metrics].to_numpy()
        bars = ax_upside.bar(x2 + offset, vals, width=width, color=quant_color(quantization), label=quantization.upper())
        for bar, value in zip(bars, vals):
            label = f"{value/100:.1f}x" if bar.get_x() < 0.5 else f"{value:.0f}%"
            ax_upside.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4, label, ha="center", va="bottom", fontsize=7, color=POSTER_COLORS["ink"], weight="bold")
    ax_upside.axhline(100, color=POSTER_COLORS["border"], linewidth=1.2, linestyle="--")
    ax_upside.set_xticks(x2, ["Throughput", "Toy-score\nretention"])
    ax_upside.set_ylim(0, max(330, float(plot_df[upside_metrics].max().max()) * 1.14))
    ax_upside.set_title("Benefits to preserve: higher bars are better", fontsize=11, weight="bold")
    ax_upside.legend(loc="upper right", ncol=3, fontsize=8, frameon=True, facecolor="#ffffff", edgecolor=POSTER_COLORS["border"])

    q4 = plot_df.loc["q4"] if "q4" in plot_df.index else plot_df.iloc[-1]
    fig.suptitle("Learning 2: same model, different precision; FP16 is the 100% baseline", fontsize=14, weight="bold", color=POSTER_COLORS["ink"])
    fig.text(
        0.5, 0.015,
        f"Takeaway: q4 averaged {q4['VRAM']:.0f}% VRAM, {q4['Energy']:.0f}% energy, {q4['Latency']:.0f}% latency, and {q4['Throughput']/100:.1f}x throughput versus FP16; toy-score retention was {q4['Quality']:.0f}%.",
        ha="center", va="bottom", fontsize=9.5, color=POSTER_COLORS["ink"], weight="bold",
        bbox=dict(facecolor="#f0f4fa", edgecolor=POSTER_COLORS["border"], boxstyle="round,pad=0.35"),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(path, dpi=190, facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_learning3_workload_specialization(summary: pd.DataFrame, path: Path) -> Path:
    gpu = primary_gpu_profile(summary)
    data = summary[(summary.hardware_profile == gpu) & (summary.params_b.between(6.0, 8.5)) & (summary.quantization == "q4")].copy()
    wanted_families = ["Gemma", "Mistral", "CodeLlama", "DeepSeek Coder", "Granite"]
    data = data[data.family.isin(wanted_families)]
    workloads = [w for w in ["code_generation", "summarization", "chat_completion", "semantic_search"] if w in set(data["workload"])]
    pivot = data.pivot_table(index="model_label", columns="workload", values="score", aggfunc="mean")
    if pivot.empty or not workloads:
        return blank(path, "Learning 3: No Single Model Wins Every Workload")
    pivot = pivot[workloads].fillna(0.0)
    order = []
    for family in wanted_families:
        matches = [idx for idx in pivot.index if idx.startswith(family.replace("DeepSeek", "DeepSeek")) or family in idx]
        order.extend(matches)
    order = [idx for idx in order if idx in pivot.index] or list(pivot.index)
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    light_ax(ax)
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), [workload_label(c) for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = float(pivot.values[i, j])
            color = "#ffffff" if value >= 0.62 else POSTER_COLORS["ink"]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8.5, weight="bold")
    ax.set_title("Workload-specialization matrix: measured toy score", fontsize=14, weight="bold", color=POSTER_COLORS["ink"], pad=10)
    ax.set_xlabel("Workload")
    ax.set_ylabel("Similar-size q4 model")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.018)
    cbar.set_label("Toy task score (higher is better)", color=POSTER_COLORS["body"], fontsize=8.5)
    cbar.ax.tick_params(colors=POSTER_COLORS["body"], labelsize=7.5)
    fig.text(0.5, 0.02, "Read horizontally: the same model can look strong on one workload and weak on another.", ha="center", fontsize=9, color=POSTER_COLORS["body"], weight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(path, dpi=190, facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_workload_active_energy(summary: pd.DataFrame, path: Path) -> Path:
    """Poster-facing view: active-device joules for common LLM workloads."""
    if summary.empty or "active_device_energy_j" not in summary:
        return blank(path, "Active Device Energy by Workload")

    workloads = [
        ("chat_completion", "Chat completion"),
        ("summarization", "Document summary"),
        ("embedding_search", "Batch embeddings"),
    ]
    available = [(workload, title) for workload, title in workloads if workload in set(summary["workload"])]
    if not available:
        return blank(path, "Active Device Energy by Workload", "Run chat, summarization, and embedding workloads to populate this panel.")

    fig, axes = plt.subplots(1, len(available), figsize=(12.2, 4.65), squeeze=False)
    axes = axes.flat
    colors = {"cpu": "#79b8b3", primary_gpu_profile(summary): "#f28e2b"}
    fallback_gpu = "#f28e2b"

    for ax, (workload, title) in zip(axes, available):
        light_ax(ax)
        data = summary[summary["workload"].eq(workload)].copy()
        if data.empty:
            ax.set_axis_off()
            continue

        gpu = primary_gpu_profile(summary)
        if workload == "embedding_search":
            # Embedding models are the meaningful comparison here; keep both measured rows if present.
            selected = data.sort_values("active_device_energy_j").head(6)
        else:
            gpu_rows = data[data["hardware_profile"].eq(gpu)].sort_values("active_device_energy_j").head(5)
            cpu_rows = data[data["hardware_profile"].eq("cpu")].sort_values("active_device_energy_j").head(1)
            selected = pd.concat([gpu_rows, cpu_rows], ignore_index=True).drop_duplicates(["model_id", "hardware_profile"])
            selected = selected.sort_values("active_device_energy_j", ascending=True).tail(6)

        if selected.empty:
            ax.set_axis_off()
            continue

        selected = selected.sort_values("active_device_energy_j", ascending=True)
        labels = [
            f"{row.model_label.replace('CodeLlama', 'CL').replace('Granite Code', 'Granite')}\n{row.hardware_profile.replace('l40s_gpu', 'GPU').replace('cpu', 'CPU')}"
            for row in selected.itertuples()
        ]
        bar_colors = [colors.get(hw, fallback_gpu) for hw in selected["hardware_profile"]]
        bars = ax.barh(labels, selected["active_device_energy_j"], color=bar_colors, alpha=0.95)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("active-device energy (J)")
        ax.tick_params(axis="y", labelsize=7.2)
        xmax = max(float(selected["active_device_energy_j"].max()), 1.0)
        ax.set_xlim(0, xmax * 1.22)
        for bar, value in zip(bars, selected["active_device_energy_j"]):
            ax.text(value + xmax * 0.025, bar.get_y() + bar.get_height() / 2, f"{value:.0f}J", va="center", ha="left", fontsize=7.5, color=POSTER_COLORS["ink"], weight="bold")
        if workload == "embedding_search":
            ax.text(0.98, 0.04, "tiny batches can favor CPU or small models", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color=POSTER_COLORS["muted"])

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#79b8b3", label="CPU-side estimate"),
        plt.Rectangle((0, 0), 1, 1, color="#f28e2b", label="GPU active NVML"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=True, facecolor="#ffffff", edgecolor=POSTER_COLORS["border"], fontsize=8.5)
    fig.suptitle("Measured active-device energy: same benchmark harness, different workloads", fontsize=14, weight="bold", color=POSTER_COLORS["ink"], y=1.0)
    fig.text(
        0.5,
        0.02,
        "Read within each panel: lower bars consume less active-device energy for that workload. This is an energy view, not a universal quality ranking.",
        ha="center",
        fontsize=9,
        color=POSTER_COLORS["body"],
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.86])
    fig.savefig(path, dpi=190, facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_learning3_cpu_gpu_tradeoff(summary: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    light_ax(ax)
    x = np.array([1, 2, 4, 8, 16, 32])
    cpu = np.array([1.0, 1.35, 2.05, 3.4, 5.8, 9.8])
    gpu = np.array([1.7, 1.35, 1.08, 0.92, 0.86, 0.84])
    ax.plot(x, cpu, marker="o", linewidth=2.8, color="#f59e0b", label="CPU-only path")
    ax.plot(x, gpu, marker="o", linewidth=2.8, color="#2563eb", label="GPU + batching path")
    ax.fill_between(x, cpu, gpu, where=cpu > gpu, color="#dcfce7", alpha=0.65)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(v) for v in x])
    ax.set_xlabel("Concurrency or batch size")
    ax.set_ylabel("Relative latency/cost pressure")
    ax.set_title("CPU vs GPU: conceptual scale inflection point", fontsize=12, weight="bold")
    ax.legend(loc="upper left", fontsize=8, frameon=True, facecolor="#ffffff", edgecolor=POSTER_COLORS["border"])
    ax.text(1.05, 8.9, "conceptual heuristic\nnot measured concurrency data", fontsize=8.5, color=POSTER_COLORS["red"], weight="bold", bbox=dict(facecolor="#fff7ed", edgecolor="#fed7aa", boxstyle="round,pad=0.3"))
    ax.text(1.05, 0.55, "tiny/local jobs:\nCPU can be fine", fontsize=8.5, color=POSTER_COLORS["body"], bbox=dict(facecolor="#f8f9fa", edgecolor=POSTER_COLORS["border"], boxstyle="round,pad=0.3"))
    ax.text(9.2, 1.25, "batching/concurrency:\nGPU amortizes overhead", fontsize=8.5, color=POSTER_COLORS["body"], bbox=dict(facecolor="#f0fdf4", edgecolor="#bbf7d0", boxstyle="round,pad=0.3"))
    ax.set_ylim(0.4, 10.5)
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


def write_lifecycle_svg(metrics: pd.DataFrame, path: Path) -> Path:
    model_id = "codellama-7b-q4"
    workload = "chat_completion"
    hardware = primary_gpu_profile(metrics)
    data = metrics[
        metrics["model_id"].eq(model_id)
        & metrics["workload"].eq(workload)
        & metrics["hardware_profile"].eq(hardware)
        & metrics.get("status", pd.Series("ok", index=metrics.index)).fillna("ok").eq("ok")
    ].copy()
    if data.empty:
        data = metrics[metrics["hardware_profile"].eq(hardware)].copy()
    row = data.mean(numeric_only=True)
    label = str(data["model_label"].dropna().iloc[0]) if "model_label" in data and not data["model_label"].dropna().empty else "CodeLlama 7B Q4"
    hw_label = str(data["hardware_label"].dropna().iloc[0]) if "hardware_label" in data and not data["hardware_label"].dropna().empty else hardware.replace("_", " ").upper()

    net_gpu_j = float(row.get("net_gpu_energy_j", row.get("active_device_energy_j", 0.0)))
    cpu_j = float(row.get("cpu_energy_j", 0.0))
    raw_gpu_j = float(row.get("gpu_energy_j", 0.0))
    total_device_j = net_gpu_j + cpu_j
    latency = float(row.get("total_latency_s", 0.0))
    gpu_util = float(row.get("gpu_util_avg_pct", 0.0))
    cpu_util = float(row.get("cpu_util_avg_pct", 0.0))
    peak_vram_gb = float(row.get("gpu_mem_peak_mb", 0.0)) / 1024.0
    tokens_j = float(row.get("active_tokens_per_joule", 0.0))
    output_tokens = float(row.get("output_tokens", 0.0))
    input_tokens = float(row.get("input_tokens", 0.0))

    stages = [
        ("Network", "NET", f"{float(row.get('network_s', 0.0)):.2f}s", "#2563eb", 74),
        ("Python", "CPU", f"{cpu_util:.0f}% CPU", "#1d4ed8", 78),
        ("Tokenize", "CPU", f"{float(row.get('tokenization_s', 0.0)):.2f}s", "#ca8a04", 82),
        ("CPU/RAM", "RAM", "prep + copy", "#0e7490", 82),
        ("VRAM", "VRAM", f"{peak_vram_gb:.1f}GB", "#7c3aed", 78),
        ("Inference", "GPU", f"{float(row.get('inference_s', 0.0)):.2f}s", "#16a34a", 92),
        ("KV cache", "VRAM", "context", "#ea580c", 78),
        ("Sampling", "GPU", f"{gpu_util:.0f}% GPU", "#be185d", 80),
        ("Post", "CPU", f"{float(row.get('postprocess_s', 0.0)):.3f}s", "#1d4ed8", 72),
        ("Response", "NET", "return", "#2563eb", 68),
    ]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 360">']
    parts.append('<rect width="980" height="360" fill="#ffffff"/>')
    parts.append(f'<text x="26" y="35" fill="#10213d" font-size="24" font-weight="800">Measured Energy Trace: {label} · Chat</text>')
    parts.append(f'<text x="26" y="60" fill="#6c7a89" font-size="13.2">One measured L40S request profile. Energy is idle-adjusted GPU NVML plus a CPU-side estimate; raw GPU draw includes idle.</text>')
    parts.append('<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#1b3a5c"/></marker></defs>')

    kpis = [
        ("Device-side energy", f"{total_device_j:.0f} J", "GPU active + CPU est.", "#f0f4fa"),
        ("Active GPU energy", f"{net_gpu_j:.0f} J", "NVML minus idle", "#f0fdf4"),
        ("CPU-side energy", f"{cpu_j:.0f} J", "RAPL/TDP estimate", "#fff7ed"),
        ("Latency", f"{latency:.2f}s", f"{input_tokens:.0f} in / {output_tokens:.0f} out tok", "#eef4ff"),
        ("Peak VRAM", f"{peak_vram_gb:.1f} GB", f"{tokens_j:.2f} tokens/J", "#f5f3ff"),
    ]
    x0 = 26
    for title, value, sub, fill in kpis:
        parts.append(f'<rect x="{x0}" y="80" width="178" height="58" rx="8" fill="{fill}" stroke="#d1d5db"/>')
        parts.append(f'<text x="{x0+12}" y="101" fill="#6c7a89" font-size="10" font-weight="800">{title}</text>')
        parts.append(f'<text x="{x0+12}" y="124" fill="#10213d" font-size="20" font-weight="900">{value}</text>')
        parts.append(f'<text x="{x0+98}" y="124" fill="#6c7a89" font-size="9">{sub}</text>')
        x0 += 188

    parts.append('<line x1="32" y1="168" x2="948" y2="168" stroke="#d1d5db" stroke-width="2"/>')
    parts.append(f'<text x="32" y="162" fill="#6c7a89" font-size="10.5" font-weight="800">REQUEST TRACE · {hw_label}</text>')

    x = 24
    y = 183
    for i, (stage, tag, hint, color, width) in enumerate(stages, start=1):
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="62" rx="8" fill="#f8f9fa" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<rect x="{x+7}" y="{y+7}" width="36" height="16" rx="8" fill="{color}" opacity=".9"/>')
        parts.append(f'<text x="{x+25}" y="{y+19}" text-anchor="middle" fill="#ffffff" font-size="8.5" font-weight="800">{tag}</text>')
        parts.append(f'<text x="{x+width/2}" y="{y+39}" text-anchor="middle" fill="#10213d" font-size="11.5" font-weight="800">{stage}</text>')
        parts.append(f'<text x="{x+width/2}" y="{y+54}" text-anchor="middle" fill="#6c7a89" font-size="8.4">{hint}</text>')
        if i < len(stages):
            parts.append(f'<path d="M{x+width+3} {y+31} H{x+width+9}" stroke="#1b3a5c" stroke-width="2" marker-end="url(#a)"/>')
        x += width + 12

    callouts = [
        (42, 274, "GPU dominates this call", f"{net_gpu_j:.0f}J active GPU vs {cpu_j:.0f}J CPU-side estimate.", "#f0fdf4", "#16a34a"),
        (330, 274, "VRAM footprint is visible", f"{label} peaked at {peak_vram_gb:.1f}GB on this run.", "#f5f3ff", "#7c3aed"),
        (622, 274, "Latency is mostly inference", f"{float(row.get('inference_s', 0.0)):.2f}s of {latency:.2f}s total latency.", "#eef4ff", "#2563eb"),
    ]
    for x1, y1, title, body, fill, stroke in callouts:
        parts.append(f'<rect x="{x1}" y="{y1}" width="270" height="48" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        parts.append(f'<text x="{x1+12}" y="{y1+19}" fill="#10213d" font-size="11.5" font-weight="800">{title}</text>')
        parts.append(f'<text x="{x1+12}" y="{y1+36}" fill="#2c3e50" font-size="9.2">{body}</text>')

    parts.append(f'<text x="26" y="344" fill="#6c7a89" font-size="10.5">Raw GPU draw: {raw_gpu_j:.0f}J. Active GPU energy subtracts estimated idle power; CPU-side energy is labeled separately.</text>')
    parts.append("</svg>")
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


def write_pycon_us_2026_logo(path: Path) -> Path:
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 348" role="img" aria-labelledby="title desc">
<title id="title">PyCon US 2026 logo</title>
<desc id="desc">A poster header SVG approximation of the supplied PyCon US 2026 Long Beach mark.</desc>
<defs>
  <linearGradient id="sun" x1="0" x2="0" y1="0" y2="1">
    <stop offset="0" stop-color="#ffe071"/>
    <stop offset="0.42" stop-color="#ff743e"/>
    <stop offset="0.72" stop-color="#ff2f85"/>
    <stop offset="1" stop-color="#a600ff"/>
  </linearGradient>
  <filter id="textGlow" x="-20%" y="-20%" width="140%" height="140%">
    <feFlood flood-color="#ff00ff" flood-opacity="0.95"/>
    <feComposite in2="SourceAlpha" operator="in"/>
    <feGaussianBlur stdDeviation="1.4"/>
    <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>
<rect width="420" height="348" fill="none"/>
<g transform="translate(34 0)">
  <circle cx="210" cy="95" r="92" fill="url(#sun)"/>
  <g stroke="#17213b" stroke-width="4" opacity=".9">
    <line x1="122" y1="103" x2="298" y2="103"/>
    <line x1="118" y1="120" x2="302" y2="120"/>
    <line x1="123" y1="137" x2="297" y2="137"/>
    <line x1="133" y1="154" x2="287" y2="154"/>
  </g>
  <text x="244" y="32" fill="#5c0099" font-family="Comic Sans MS, Segoe Print, cursive" font-size="16" transform="rotate(32 244 32)">Long Beach</text>
  <g fill="#4b007d" stroke="#4b007d" stroke-width="2">
    <path d="M176 173 C186 118 199 77 205 43 C213 86 210 124 198 180 Z"/>
    <path d="M234 169 C240 129 250 99 258 70 C262 108 258 143 249 177 Z"/>
    <path d="M205 46 C186 31 166 29 148 40 C172 40 188 46 205 62 Z"/>
    <path d="M205 49 C221 31 245 28 264 38 C238 40 222 48 205 66 Z"/>
    <path d="M205 51 C198 30 205 14 223 2 C216 25 213 42 207 67 Z"/>
    <path d="M258 72 C241 58 223 57 208 66 C228 67 243 73 257 86 Z"/>
    <path d="M258 74 C274 57 292 55 308 65 C289 65 273 75 260 91 Z"/>
    <path d="M258 75 C255 56 263 42 280 34 C271 53 266 67 260 92 Z"/>
  </g>
  <path d="M78 178 C120 178 125 226 169 225 C209 224 219 177 259 177 C298 177 307 226 346 225" fill="none" stroke="#ff43d2" stroke-width="42" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M78 166 C120 166 125 214 169 213 C209 212 219 165 259 165 C298 165 307 214 346 213" fill="none" stroke="#1ddff2" stroke-width="38" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="78" cy="166" r="3" fill="#17213b"/>
  <circle cx="346" cy="213" r="3" fill="#17213b"/>
</g>
<text x="210" y="282" text-anchor="middle" fill="#ffffff" stroke="#ff00ff" stroke-width="7" paint-order="stroke" filter="url(#textGlow)" font-family="Inter, Arial Black, sans-serif" font-size="58" font-weight="900" letter-spacing="1">PYCON.US</text>
<text x="210" y="330" text-anchor="middle" fill="#ffffff" stroke="#ff00ff" stroke-width="6" paint-order="stroke" filter="url(#textGlow)" font-family="Inter, Arial Black, sans-serif" font-size="48" font-weight="900">2026</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")
    return path


def build_story_poster(results_dir: Path, assets: Path, path: Path) -> Path:
    metrics_path = results_dir / "metrics.csv"
    summary_path = results_dir / "summary_enriched.csv"
    hardware_path = results_dir / "hardware.json"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    rows_count = int(len(metrics))
    model_count = int(metrics["model_id"].nunique()) if "model_id" in metrics else 0
    workload_count = int(metrics["workload"].nunique()) if "workload" in metrics else 0
    hardware_name = "NVIDIA L40S"
    if hardware_path.exists():
        try:
            hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
            if hardware.get("gpus"):
                hardware_name = hardware["gpus"][0].get("name", hardware_name)
        except json.JSONDecodeError:
            pass
    pycon_logo = write_pycon_us_2026_logo(assets / "pycon-us-2026-logo.svg")
    qr_source = Path(__file__).resolve().parents[1] / "poster" / "system_assets" / "repo_qr.png"
    qr_asset = assets / "repo_qr.png"
    if qr_source.exists():
        shutil.copyfile(qr_source, qr_asset)
    qr_markup = (
        '<img class="qr-img" src="repo_qr.png" alt="QR code for the open repository">'
        if qr_asset.exists()
        else '<div class="qr-slot">QR<br>repo</div>'
    )

    def cpu_gpu_rows() -> str:
        if summary.empty:
            return '<tr><td colspan="3">Run CPU/GPU benchmarks to populate this table.</td></tr>'
        data = summary[summary.model_id.eq("codellama-7b-q4")]
        gpu = primary_gpu_profile(summary)
        rows = []
        for workload in ["chat_completion", "summarization", "code_generation", "semantic_search"]:
            sub = data[data.workload.eq(workload)]
            cpu = sub[sub.hardware_profile.eq("cpu")]
            gpu_row = sub[sub.hardware_profile.eq(gpu)]
            if cpu.empty or gpu_row.empty:
                continue
            c = cpu.iloc[0]
            g = gpu_row.iloc[0]
            speedup = c["total_latency_s"] / max(g["total_latency_s"], 1e-9)
            rows.append(
                f"<tr><td>{workload_label(workload)}</td><td>{c['total_latency_s']:.1f}s CPU → <strong>{g['total_latency_s']:.1f}s GPU</strong></td><td class=\"best\">{speedup:.1f}x faster</td></tr>"
            )
        return "".join(rows) or '<tr><td colspan="3">CPU/GPU comparison rows unavailable.</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cost &amp; Energy of Everyday LLM Workloads: A Visual Field Guide for Python Developers</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --ink:#1a1a2e; --heading:#0f1b35; --body:#2c3e50; --muted:#6c7a89;
      --accent:#1b3a5c; --border:#d1d5db; --bg:#ffffff; --surface:#f8f9fa;
      --green:#1a7a3a; --orange:#d97706; --blue:#1b3a5c;
    }}
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ background:#e5e7eb; font-family:"Inter",system-ui,sans-serif; color:var(--body); -webkit-font-smoothing:antialiased; }}
    code {{ font-family:"JetBrains Mono",monospace; font-size:.88em; background:#f0f0f0; padding:1px 4px; border-radius:2px; }}
    .poster {{ width:min(100vw - 16px,1580px); aspect-ratio:84.1/59.4; margin:8px auto; background:var(--bg); box-shadow:0 4px 20px rgba(0,0,0,.12); overflow:hidden; display:flex; flex-direction:column; }}
    @page {{ size:84.1cm 59.4cm landscape; margin:0; }}
    @media print {{
      body {{ background:#fff; margin:0; padding:0; }}
      .poster {{ width:84.1cm; height:59.4cm; margin:0; box-shadow:none; aspect-ratio:auto; }}
      .poster * {{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
    }}
    .header {{ background:#1a2744; padding:6px 22px; display:flex; align-items:center; gap:16px; color:#fff; flex-shrink:0; }}
    .logo-area {{ flex:0 0 auto; display:flex; align-items:center; gap:10px; min-width:198px; }}
    .header-qr {{ flex:0 0 198px; display:flex; justify-content:flex-end; align-items:center; }}
    .py-box {{ width:52px; height:52px; border:1.5px solid rgba(255,255,255,.55); border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:800; }}
    .pycon-logo {{ width:78px; height:64px; object-fit:contain; display:block; }}
    .conf-label {{ font-size:13px; font-weight:700; line-height:1.25; }}
    .conf-label small {{ display:block; font-size:9px; font-weight:500; opacity:.8; text-transform:uppercase; letter-spacing:.08em; }}
    .center {{ flex:1; text-align:center; min-width:0; }}
    h1 {{ font-size:clamp(16px,1.7vw,24px); line-height:1.12; font-weight:800; letter-spacing:-.01em; }}
    .authors {{ font-size:10.5px; margin-top:4px; opacity:.95; font-weight:600; }}
    .affiliations {{ font-size:9px; margin-top:2px; opacity:.78; }}
    .glance-row {{ display:grid; grid-template-columns:132px repeat(4,1fr); gap:8px; padding:5px 16px; border-bottom:1.5px solid var(--border); background:#fbfcff; flex-shrink:0; align-items:stretch; }}
    .glance-label {{ display:flex; align-items:center; color:var(--accent); font-size:12px; line-height:1.05; font-weight:800; text-transform:uppercase; letter-spacing:.05em; }}
    .glance-card {{ border:1px solid var(--border); border-radius:5px; background:#ffffff; padding:6px 10px; min-height:52px; }}
    .glance-card b {{ display:block; color:var(--heading); font-size:13.6px; line-height:1.05; margin-bottom:3px; }}
    .glance-card span {{ display:block; color:var(--body); font-size:9px; line-height:1.24; }}
    .repo-note {{ background:#1a2744; color:#cfd8e6; border-top:1px solid #33415f; padding:4px 16px; font-size:7.8px; line-height:1.2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex-shrink:0; }}
    .body {{ display:grid; grid-template-columns:.98fr 1.08fr .98fr; flex:1; min-height:0; }}
    .col {{ padding:11px 15px 11px; border-right:1.5px solid var(--border); overflow:hidden; }}
    .col:last-child {{ border-right:none; }}
    .section {{ margin-bottom:9px; }}
    .section-title {{ font-size:12.8px; font-weight:800; text-transform:uppercase; color:var(--accent); letter-spacing:.04em; padding-bottom:4px; border-bottom:2px solid var(--accent); margin-bottom:6px; }}
    .section-title.blue {{ color:var(--blue); border-bottom-color:var(--blue); }}
    .badge {{ display:inline-flex; align-items:center; gap:4px; background:#eef4ff; border:1px solid #c7d7f2; color:#1b3a5c; border-radius:999px; padding:3px 8px; font-size:8.1px; font-weight:800; letter-spacing:.05em; text-transform:uppercase; margin-bottom:5px; }}
    .learning-title {{ font-size:14.6px; font-weight:800; color:var(--heading); line-height:1.15; margin-bottom:4px; }}
    .tagline {{ font-size:10.6px; font-weight:800; color:var(--orange); line-height:1.25; margin-bottom:5px; }}
    p, li {{ font-size:9.8px; line-height:1.38; color:var(--body); }}
    strong {{ color:var(--ink); }}
    ul {{ padding-left:14px; margin:4px 0; }}
    li {{ margin-bottom:2px; }}
    table {{ width:100%; border-collapse:collapse; font-size:8.25px; margin:5px 0; }}
    th, td {{ border:1px solid var(--border); padding:2.6px 4.6px; text-align:center; }}
    th {{ background:var(--surface); color:var(--heading); font-weight:700; }}
    td:first-child {{ text-align:left; font-weight:700; }}
    .best {{ color:var(--green); font-weight:800; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin:7px 0; }}
    .kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:4px; padding:7px; min-height:52px; }}
    .kpi b {{ display:block; color:var(--accent); font-size:21px; line-height:1; margin-bottom:4px; }}
    .kpi span {{ display:block; color:var(--muted); font-size:8.6px; line-height:1.22; }}
    .figure {{ border:1.5px solid var(--border); border-radius:4px; margin:6px 0; overflow:hidden; }}
    .fig-caption {{ background:var(--surface); padding:5px 10px; font-size:9.4px; font-weight:800; color:var(--muted); border-bottom:1px solid var(--border); display:flex; justify-content:space-between; gap:8px; }}
    .fig-body {{ padding:6px; }}
    .chart {{ width:100%; display:block; object-fit:contain; }}
    .chart-life {{ height:152px; }}
    .chart-large {{ height:276px; }}
    .chart-xl {{ height:246px; }}
    .chart-mid {{ height:205px; }}
    .chart-small {{ height:128px; }}
    .callout {{ background:#f0f4fa; border-left:3px solid var(--accent); padding:5px 8px; font-size:9.8px; line-height:1.32; margin:5px 0; border-radius:0 3px 3px 0; }}
    .trace-note {{ font-size:8.8px; line-height:1.25; padding:5px 8px; margin-top:5px; }}
    .caution {{ background:#fff7ed; border-left:3px solid var(--orange); }}
    .score-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:5px; margin-top:5px; }}
    .score-card {{ background:#f8f9fa; border:1px solid var(--border); border-radius:4px; padding:6px; min-height:54px; }}
    .score-card b {{ display:block; color:var(--heading); font-size:8.8px; margin-bottom:2px; }}
    .score-card span {{ display:block; color:var(--body); font-size:7.8px; line-height:1.25; }}
    .choice-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:7px 0; }}
    .choice-box {{ background:#f8f9fa; border:1px solid var(--border); border-radius:4px; padding:6px 7px; min-height:64px; }}
    .choice-box b {{ display:block; color:var(--heading); font-size:10px; margin-bottom:4px; }}
    .choice-box ul {{ margin:0; padding-left:13px; }}
    .choice-box li {{ font-size:8.2px; line-height:1.2; margin-bottom:0; }}
    .decision {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; margin-top:6px; }}
    .decision div {{ background:#f8f9fa; border:1px solid var(--border); border-radius:4px; padding:6px; min-height:58px; }}
    .decision b {{ display:block; color:var(--heading); font-size:9px; margin-bottom:3px; }}
    .stack {{ display:flex; flex-direction:column; }}
    .stack-row {{ display:flex; border:1px solid var(--border); border-bottom:none; font-size:9.6px; line-height:1.34; }}
    .stack-row:last-child {{ border-bottom:1px solid var(--border); }}
    .stack-tag {{ flex:0 0 82px; display:flex; align-items:center; justify-content:center; text-align:center; padding:5px 6px; font-size:7.4px; font-weight:800; letter-spacing:.06em; text-transform:uppercase; }}
    .stack-desc {{ flex:1; padding:5px 7px; }}
    .tag-fixtures {{ background:#dbeafe; color:#1e3a8a; }}
    .tag-client {{ background:#dcfce7; color:#14532d; }}
    .tag-energy {{ background:#fff7ed; color:#9a3412; }}
    .tag-visual {{ background:#f3e8ff; color:#581c87; }}
    .note {{ font-size:8.7px; color:var(--muted); line-height:1.34; margin-top:4px; }}
    .qr-slot {{ width:60px; height:60px; background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.48); border-radius:4px; display:flex; align-items:center; justify-content:center; font-size:7px; font-weight:800; color:rgba(255,255,255,.82); text-transform:uppercase; text-align:center; }}
    .qr-img {{ width:60px; height:60px; object-fit:contain; display:block; background:#fff; padding:3px; border-radius:4px; border:1px solid rgba(255,255,255,.55); }}
  </style>
</head>
<body>
<div class="poster" contenteditable="false">
  <div class="header">
    <div class="logo-area"><img class="pycon-logo" src="{pycon_logo.name}" alt="PyCon US 2026 logo"><div class="conf-label">PyCon US<br><small>Poster Session</small></div></div>
    <div class="center">
      <h1>Cost &amp; Energy of Everyday LLM Workloads: A Visual Field Guide for Python Developers</h1>
      <div class="authors">Shivay Lamba &nbsp;&nbsp; Suvrakamal Das</div>
      <div class="affiliations">shivaylamba@gmail.com &nbsp;&nbsp; subhrokomol@gmail.com</div>
    </div>
    <div class="header-qr">{qr_markup}</div>
  </div>

  <div class="glance-row">
    <div class="glance-label">Results<br>at a Glance</div>
    <div class="glance-card"><b>Bigger ≠ Better</b><span>Scaling model size often raises energy, latency, and VRAM faster than toy-task quality.</span></div>
    <div class="glance-card"><b>Quantization Changes Economics</b><span>q4/q8 variants can cut memory and energy enough to change where models can run.</span></div>
    <div class="glance-card"><b>No Single Winner</b><span>The best model depends on workload: code, summary, chat, RAG, or embeddings.</span></div>
    <div class="glance-card"><b>Hardware Depends on Scale</b><span>CPU can be fine for tiny local jobs; GPU wins when batching or concurrency grows.</span></div>
  </div>
  <div class="body">
    <div class="col">
      <div class="section">
        <div class="section-title">Overview</div>
        <p>Everyday LLM calls look simple from Python, but the real deployment decision is a tradeoff between <strong>quality</strong>, <strong>latency</strong>, <strong>VRAM</strong>, <strong>energy</strong>, and <strong>cost</strong>.</p>
        <p style="margin-top:4px;">We benchmark chat completions, document summarization, classification-style prompts, RAG-style search, code generation, and batch embeddings across model sizes, quantization levels, architectures, and CPU/GPU configurations.</p>
        <div class="callout"><strong>Thesis:</strong> the most accurate LLM is often not the most economically efficient one. LLM deployment is a systems engineering problem, not just model selection.</div>
      </div>

      <div class="section">
        <div class="section-title">Benchmark Scope</div>
        <div class="kpis">
          <div class="kpi"><b>{model_count}</b><span>models / variants</span></div>
          <div class="kpi"><b>{workload_count}</b><span>workload families</span></div>
          <div class="kpi"><b>{rows_count}</b><span>metric rows</span></div>
          <div class="kpi"><b>L40S</b><span>{hardware_name}</span></div>
        </div>
        <table>
          <thead><tr><th>Question</th><th>Measured Signals</th><th>Why It Matters</th></tr></thead>
          <tbody>
            <tr><td>Bigger model?</td><td>toy score, latency, joules, VRAM</td><td>find diminishing returns</td></tr>
            <tr><td>Quantize?</td><td>q4/q8/fp16, tokens/J</td><td>fit smaller hardware</td></tr>
            <tr><td>Which workload?</td><td>latency slices, joules, cost, score</td><td>avoid one-model thinking</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <div class="section-title blue">Measurement Pipeline</div>
        <p>Each run produces one row of metrics: latency slices, token counts, active-device energy, memory, workload score, and request-cost estimates.</p>
        <div class="stack">
          <div class="stack-row"><div class="stack-tag tag-fixtures">Fixtures</div><div class="stack-desc">Repeatable prompts and document sets for each workload.</div></div>
          <div class="stack-row"><div class="stack-tag tag-client">Client</div><div class="stack-desc">Ollama/OpenAI-compatible adapter wraps model calls and response logs.</div></div>
          <div class="stack-row"><div class="stack-tag tag-energy">Energy</div><div class="stack-desc">NVML samples GPU power/utilization/VRAM; CPU uses RAPL or labeled TDP estimate.</div></div>
          <div class="stack-row"><div class="stack-tag tag-visual">Visuals</div><div class="stack-desc">pandas + matplotlib generate latency waterfalls, cost bars, and active-energy figures.</div></div>
        </div>
        <div class="callout caution"><strong>Energy attribution matters:</strong> active-device energy is cleaner for model comparison; total wall energy is better for full-machine cost.</div>
      </div>

      <div class="section">
        <div class="section-title">What Gets Timed</div>
        <table>
          <thead><tr><th>Slice</th><th>Meaning</th><th>Why Python Devs Care</th></tr></thead>
          <tbody>
            <tr><td>Network</td><td>client call overhead</td><td>endpoint placement, batching</td></tr>
            <tr><td>Tokenization</td><td>prompt processing / prefill</td><td>long docs are not free</td></tr>
            <tr><td>Inference</td><td>model generation</td><td>model size dominates</td></tr>
            <tr><td>Post-processing</td><td>parse, serialize, rank</td><td>pipeline glue is measurable</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <div class="section-title">Measured Energy Trace for One LLM</div>
        <div class="figure">
          <div class="fig-caption"><span>Measured example: CodeLlama 7B Q4 chat on L40S</span><span>CPU + RAM + GPU + VRAM</span></div>
          <div class="fig-body"><img class="chart chart-life" src="request_lifecycle.svg" alt="Measured energy trace"></div>
        </div>
        <div class="callout trace-note"><strong>How to read it:</strong> one measured CodeLlama 7B Q4 chat request. The cards show active-device joules, latency, and peak VRAM; the trace locates Python client work, prefill, GPU inference, KV cache, sampling, and response parsing.</div>
      </div>
    </div>

    <div class="col">
      <div class="section">
        <div class="section-title">Learning 1: Bigger Models Have Diminishing Returns</div>
        <div class="badge">SAME ARCHITECTURE · DIFFERENT SIZES</div>
        <div class="tagline">Scaling parameters often increases infrastructure cost faster than model quality.</div>
        <p class="note">Same-family comparisons isolate parameter count: Gemma 2B/7B, Phi3 3.8B/14B, Granite 3B/8B/20B, CodeLlama 7B/13B where present.</p>
        <div class="figure">
          <div class="fig-caption"><span>Measured benchmark: toy/code score, energy, latency, and VRAM</span><span>SAME-FAMILY SCALING</span></div>
          <div class="fig-body"><img class="chart chart-large" src="learning1_scaling_tradeoffs.png" alt="Scaling tradeoffs"></div>
        </div>
        <div class="callout"><strong>Score note:</strong> the quality panel is a tiny code-generation pass rate, not a claim that lower-parameter models are universally smarter. Developer lesson: bigger is not automatically cheaper, faster, more deployable, or more practical.</div>
      </div>

      <div class="section">
        <div class="section-title">Learning 2: Quantization Changes the Economics of AI</div>
        <div class="badge">SAME MODEL · DIFFERENT PRECISION</div>
        <div class="tagline">4-bit and 8-bit models often retain useful quality while reducing memory, energy, and latency.</div>
        <div class="figure">
          <div class="fig-caption"><span>Measured benchmark: fp16 vs q8 vs q4 only within the same model</span><span>% OF SAME-MODEL FP16 BASELINE</span></div>
          <div class="fig-body"><img class="chart chart-xl" src="learning2_quantization_tradeoffs.png" alt="Quantization tradeoffs"></div>
        </div>
        <div class="callout"><strong>How to read it:</strong> FP16 is the 100% baseline for the exact same model; q8/q4 are compressed representations. Left: costs to minimize (VRAM, energy, latency). Right: benefits to preserve or improve (throughput, toy-score retention). Bars average only complete same-model FP16/q8/q4 triplets.</div>
      </div>
    </div>

    <div class="col">
      <div class="section">
        <div class="section-title">Learning 3: Energy Depends on the Workload</div>
        <div class="badge">COMMON WORKLOADS · ACTIVE-DEVICE JOULES</div>
        <div class="tagline">Chat, document summaries, RAG, and embeddings do not stress the same parts of the stack.</div>
        <div class="figure">
          <div class="fig-caption"><span>Measured benchmark: active-device energy by scenario</span><span>JOULES / REQUEST · LOWER IS BETTER</span></div>
          <div class="fig-body"><img class="chart chart-mid" src="workload_active_device_energy.png" alt="Active device energy for chat, summarization, and embeddings"></div>
        </div>
        <div class="callout"><strong>How to read it:</strong> each panel is a different workload. A model/hardware pair that is cheap for chat may not be cheap for long summaries or embedding batches, so the benchmark must be repeated per scenario.</div>
      </div>

      <div class="section">
        <div class="section-title blue">CPU vs GPU: The Scale Inflection Point</div>
        <div class="badge">HARDWARE SCALE · CONCEPTUAL HEURISTIC</div>
        <p class="note">Hardware choice changes when prompts, outputs, batching, or concurrency grow. Single-request latency is not the same as throughput.</p>
        <div class="figure">
          <div class="fig-caption"><span>Conceptual deployment heuristic, not measured concurrency data</span><span>CPU-ONLY VS GPU + BATCHING</span></div>
          <div class="fig-body"><img class="chart chart-small" src="learning3_cpu_gpu_tradeoff.png" alt="CPU GPU tradeoff"></div>
        </div>
        <div class="choice-grid">
          <div class="choice-box"><b>CPU is reasonable for</b><ul><li>prototyping</li><li>local scripts</li><li>tiny embedding jobs</li><li>single-user offline tools</li><li>low request volume</li></ul></div>
          <div class="choice-box"><b>GPU becomes better for</b><ul><li>long prompts / outputs</li><li>batch embeddings</li><li>multi-user chat</li><li>latency-sensitive services</li><li>high concurrency</li></ul></div>
        </div>
        <p class="note"><strong>Tiny embedding jobs may not amortize GPU overhead.</strong> Measure batch size before assuming the GPU path wins.</p>
      </div>

      <div class="section">
        <div class="section-title">Practical Heuristics</div>
        <table>
          <thead><tr><th>#</th><th>Heuristic</th></tr></thead>
          <tbody>
            <tr><td>1</td><td>Benchmark the workload, not the model card.</td></tr>
            <tr><td>2</td><td>If quality barely improves but joules rise, use the smaller model.</td></tr>
            <tr><td>3</td><td>If VRAM is the deployment constraint, use q4/q8.</td></tr>
            <tr><td>4</td><td>If the workload changes, re-benchmark; do not reuse one winner.</td></tr>
            <tr><td>5</td><td>If concurrency grows, use GPU batching.</td></tr>
            <tr><td>6</td><td>If the workload is tiny and local, CPU may be good enough.</td></tr>
            <tr><td>7</td><td>Track tokens/joule, not just accuracy.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
  <div class="repo-note">Open repository: github.com/shivaylamba/pycon-2026-poster | Inspired by: Alizadeh et al., Language Models in Software Development Tasks: An Experimental Analysis of Energy and Accuracy, arXiv:2412.00329.</div>
</div>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    return path


def build_poster(results_dir: Path, assets: Path, path: Path) -> Path:
    metrics_path = results_dir / "metrics.csv"
    summary_path = results_dir / "summary_enriched.csv"
    registry_path = results_dir / "model_registry.csv"
    hardware_path = results_dir / "hardware.json"

    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()

    rows_count = int(len(metrics))
    model_count = int(metrics["model_id"].nunique()) if "model_id" in metrics else 0
    workload_count = int(metrics["workload"].nunique()) if "workload" in metrics else 0
    hardware_name = "NVIDIA L40S"
    if hardware_path.exists():
        try:
            hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
            if hardware.get("gpus"):
                hardware_name = hardware["gpus"][0].get("name", hardware_name)
        except json.JSONDecodeError:
            pass

    def selected_cpu_gpu_rows() -> str:
        if summary.empty:
            return '<tr><td colspan="4">Run benchmarks to populate this table.</td></tr>'
        data = summary[summary["model_id"].eq("codellama-7b-q4")]
        rows = []
        labels = {
            "chat_completion": "CodeLlama 7B chat",
            "summarization": "CodeLlama 7B summarize",
            "semantic_search": "CodeLlama 7B RAG query",
            "code_generation": "CodeLlama 7B code",
        }
        for workload, label in labels.items():
            sub = data[data["workload"].eq(workload)]
            cpu = sub[sub["hardware_profile"].eq("cpu")]
            gpu = sub[sub["hardware_profile"].ne("cpu")]
            if cpu.empty or gpu.empty:
                continue
            cpu_row = cpu.iloc[0]
            gpu_row = gpu.iloc[0]
            speedup = cpu_row["total_latency_s"] / max(gpu_row["total_latency_s"], 1e-9)
            rows.append(
                f"<tr><td>{label}</td><td>{cpu_row['total_latency_s']:.2f}s / {cpu_row['active_device_energy_j']:.0f}J</td>"
                f"<td class=\"best\">{gpu_row['total_latency_s']:.2f}s / {gpu_row['active_device_energy_j']:.0f}J</td>"
                f"<td>{speedup:.1f}x faster on GPU</td></tr>"
            )
        return "".join(rows) or '<tr><td colspan="4">CPU/GPU comparison not available.</td></tr>'

    def top_efficiency_rows() -> str:
        if summary.empty or "active_tokens_per_joule" not in summary:
            return '<tr><td colspan="4">Run benchmarks to populate this table.</td></tr>'
        data = summary.sort_values("active_tokens_per_joule", ascending=False).head(5)
        rows = []
        for _, row in data.iterrows():
            rows.append(
                f"<tr><td>{row['model_label']}</td><td>{row['workload'].replace('_', ' ')}</td>"
                f"<td class=\"best\">{row['active_tokens_per_joule']:.2f}</td><td>{row['tokens_per_sec']:.1f}</td></tr>"
            )
        return "".join(rows)

    if registry.empty:
        model_labels = sorted(metrics["model_label"].dropna().unique().tolist()) if "model_label" in metrics else []
    else:
        model_labels = registry.sort_values(["family", "params_b", "quantization"])["label"].dropna().unique().tolist()
    chips = "".join(f"<span>{label}</span>" for label in model_labels[:19])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cost, Energy & Infrastructure Tradeoffs — Poster</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
  :root {{
    --ink: #1a1a2e;
    --heading: #0f1b35;
    --body: #2c3e50;
    --muted: #6c7a89;
    --accent: #1b3a5c;
    --accent-light: #2a5580;
    --blue: #1b3a5c;
    --border: #d1d5db;
    --bg: #ffffff;
    --surface: #f8f9fa;
    --green: #1a7a3a;
    --red: #c0392b;
    --orange: #d97706;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #e5e7eb; font-family: "Inter", system-ui, sans-serif; color: var(--body); -webkit-font-smoothing: antialiased; }}
  code {{ font-family: "JetBrains Mono", monospace; font-size: .88em; background: #f0f0f0; padding: 1px 4px; border-radius: 2px; }}
  .poster {{ width: min(100vw - 16px, 1580px); aspect-ratio: 84.1 / 59.4; margin: 8px auto; background: var(--bg); box-shadow: 0 4px 20px rgba(0,0,0,.12); overflow: hidden; display: flex; flex-direction: column; }}
  @page {{ size: 84.1cm 59.4cm landscape; margin: 0; }}
  @media print {{
    body {{ background: #fff; margin: 0; padding: 0; }}
    .poster {{ width: 84.1cm; height: 59.4cm; margin: 0; box-shadow: none; aspect-ratio: auto; }}
    .poster * {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  }}
  .header {{ background: #1a2744; padding: 12px 24px; display: flex; align-items: center; gap: 16px; color: #fff; position: relative; overflow: hidden; flex-shrink: 0; }}
  .header .logo-area {{ flex: 0 0 auto; display: flex; align-items: center; gap: 10px; z-index: 1; min-width: 210px; }}
  .py-box {{ width: 52px; height: 52px; border: 1.5px solid rgba(255,255,255,.55); border-radius: 6px; display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:800; }}
  .conf-label {{ font-size: 13px; font-weight: 700; line-height: 1.25; letter-spacing: .01em; }}
  .conf-label small {{ display: block; font-size: 9px; font-weight: 500; opacity: .8; text-transform: uppercase; letter-spacing:.08em; }}
  .header .center {{ flex: 1; text-align: center; z-index: 1; min-width: 0; }}
  .header h1 {{ font-size: clamp(14px, 1.48vw, 20px); font-weight: 800; line-height: 1.18; letter-spacing: -.01em; }}
  .header .authors {{ font-size: 10.5px; margin-top: 4px; opacity: .95; font-weight: 600; }}
  .header .affiliations {{ font-size: 9px; margin-top: 2px; opacity: .78; }}
  .body {{ display: grid; grid-template-columns: 1fr 1.12fr 1fr; gap: 0; padding: 0; flex: 1; min-height: 0; }}
  .col {{ padding: 10px 14px 12px; border-right: 1.5px solid var(--border); overflow: hidden; }}
  .col:last-child {{ border-right: none; }}
  .section {{ margin-bottom: 6px; }}
  .section:last-child {{ margin-bottom: 0; }}
  .section-title {{ font-size: 12px; font-weight: 800; text-transform: uppercase; color: var(--accent); letter-spacing: .04em; padding-bottom: 3px; border-bottom: 2px solid var(--accent); margin-bottom: 6px; }}
  .section-title.blue {{ color: var(--blue); border-bottom-color: var(--blue); }}
  h3 {{ font-size: 11px; font-weight: 700; color: var(--heading); margin: 6px 0 3px; }}
  h3:first-child {{ margin-top: 0; }}
  p, li {{ font-size: 9.2px; line-height: 1.36; color: var(--body); }}
  ul, ol {{ padding-left: 14px; margin: 3px 0 6px; }}
  li {{ margin-bottom: 2px; }}
  strong {{ color: var(--ink); }}
  .figure {{ border: 1.5px solid var(--border); border-radius: 4px; margin: 4px 0; overflow: hidden; }}
  .fig-caption {{ background: var(--surface); padding: 4px 10px; font-size: 9px; font-weight: 700; color: var(--muted); border-bottom: 1px solid var(--border); display:flex; justify-content:space-between; gap:8px; }}
  .fig-body {{ padding: 5px; }}
  .fig-body.tight {{ padding: 3px 5px; }}
  .chart {{ width: 100%; object-fit: contain; display: block; }}
  .chart-xs {{ height: 68px; }}
  .chart-sm {{ height: 90px; }}
  .chart-md {{ height: 118px; }}
  .chart-lg {{ height: 142px; }}
  .chart-xl {{ height: 166px; }}
  .stack {{ display: flex; flex-direction: column; gap: 0; }}
  .stack-row {{ display: flex; align-items: stretch; border: 1px solid var(--border); border-bottom: none; font-size: 9.5px; line-height: 1.35; }}
  .stack-row:last-child {{ border-bottom: 1px solid var(--border); }}
  .stack-tag {{ flex: 0 0 78px; min-width: 78px; max-width: 78px; padding: 4px 5px; font-size: 7px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; display: flex; align-items: center; justify-content: center; text-align: center; }}
  .stack-desc {{ flex: 1; padding: 4px 6px; }}
  .tag-fixtures {{ background: #dbeafe; color: #1e3a8a; }}
  .tag-client {{ background: #dcfce7; color: #14532d; }}
  .tag-energy {{ background: #fff7ed; color: #9a3412; }}
  .tag-visual {{ background: #f3e8ff; color: #581c87; }}
  .algo-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 6px 8px; font-size: 9.5px; line-height: 1.4; margin: 6px 0; }}
  .algo-box .algo-title {{ font-weight: 700; font-size: 9.5px; color: var(--heading); margin-bottom: 4px; text-transform: uppercase; letter-spacing: .04em; }}
  .algo-box code {{ background: transparent; padding: 0; }}
  table {{ width: 100%; font-size: 8px; border-collapse: collapse; margin: 4px 0; }}
  th, td {{ border: 1px solid var(--border); padding: 2.4px 4px; text-align: center; }}
  th {{ background: var(--surface); font-weight: 700; font-size: 7.8px; color: var(--heading); }}
  td:first-child {{ text-align: left; font-weight: 600; }}
  .best {{ font-weight: 700; color: var(--green); }}
  .dim {{ color: var(--muted); }}
  .chips {{ display:flex; flex-wrap:wrap; gap:3px; margin: 4px 0 3px; }}
  .chips span {{ border:1px solid var(--border); background:#fff8d8; border-radius:12px; padding:2px 6px; font:700 7px "JetBrains Mono",monospace; color:#1f2937; }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:5px; margin:6px 0; }}
  .kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:4px; padding:6px; min-height:52px; }}
  .kpi b {{ display:block; color:var(--accent); font-size:18px; line-height:1; margin-bottom:3px; }}
  .kpi span {{ display:block; color:var(--muted); font-size:8px; line-height:1.25; }}
  .callout {{ background: #f0f4fa; border-left: 3px solid var(--accent); padding: 4px 7px; font-size: 8.8px; line-height: 1.28; margin: 4px 0; border-radius: 0 3px 3px 0; }}
  .note {{ font-size: 8px; color: var(--muted); line-height: 1.28; margin-top: 3px; }}
  .footer {{ border-top: 1.5px solid var(--border); padding: 6px 16px; display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-shrink: 0; }}
  .footer .refs {{ font-size: 8.5px; line-height: 1.35; color: var(--muted); flex: 1; }}
  .footer .refs strong {{ color: var(--heading); font-size: 9px; }}
  .footer .refs a {{ color: var(--blue); text-decoration: none; }}
  .footer .qr-slot {{ flex: 0 0 auto; width: 60px; height: 60px; background: var(--surface); border: 1px solid var(--border); border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 7px; font-weight: 700; color: var(--muted); text-transform: uppercase; text-align:center; }}
  .conf-bar {{ background: var(--heading); color: #8899aa; font-size: 8.5px; padding: 4px 16px; display: flex; justify-content: space-between; flex-shrink: 0; }}
  .conf-bar a {{ color: #8ecae6; text-decoration: none; }}
  hr.sec-sep {{ border: none; border-top: 1px solid var(--border); margin: 5px 0; }}
</style>
</head>
<body>
<div class="poster" contenteditable="false">
  <div class="header">
    <div class="logo-area">
      <div class="py-box">Py</div>
      <div class="conf-label">PyCon US<br><small>Poster Session</small></div>
    </div>
    <div class="center">
      <h1>Cost, Energy &amp; Infrastructure Tradeoffs of Everyday LLM Workloads:<br>A Visual Python Field Guide</h1>
      <div class="authors">Shivay Lamba &nbsp;&nbsp; Suvrakamal Das</div>
      <div class="affiliations">Python scripts, notebooks, Ollama, Hugging Face models, NVML, pandas, matplotlib, Plotly</div>
    </div>
    <div class="logo-area" style="visibility:hidden;">
      <div class="py-box">Py</div>
      <div class="conf-label">PyCon US<br><small>Poster Session</small></div>
    </div>
  </div>

  <div class="body">
    <div class="col">
      <div class="section">
        <div class="section-title">Overview</div>
        <p>
          Generative AI is everywhere, but the cost and energy impact of everyday LLM calls is usually hidden in dashboards, invoices, and GPU logs.
          This poster turns those numbers into a reproducible Python field guide for choosing local models and hardware.
        </p>
        <p style="margin-top:4px;">
          We profile <strong>code generation</strong>, <strong>document summarization</strong>, <strong>chat completion</strong>, <strong>semantic/RAG search</strong>, and <strong>batch embeddings</strong> across model sizes, quantization levels, architectures, and CPU/GPU configurations.
        </p>
        <div class="callout"><strong>Thesis:</strong> the most accurate LLM is often not the most economically efficient one. LLM deployment is a systems engineering problem, not just a model selection problem.</div>
      </div>

      <div class="section">
        <div class="section-title">Benchmark Scope</div>
        <div class="kpis">
          <div class="kpi"><b>{model_count}</b><span>models / variants</span></div>
          <div class="kpi"><b>{workload_count}</b><span>workload families</span></div>
          <div class="kpi"><b>{rows_count}</b><span>raw metric rows</span></div>
          <div class="kpi"><b>{hardware_name}</b><span>datacenter GPU run</span></div>
        </div>
        <table>
          <thead><tr><th>Dimension</th><th>Isolated Effect</th><th>Examples</th></tr></thead>
          <tbody>
            <tr><td>Model size</td><td>parameter count</td><td>Gemma 2B/7B, Phi-3 3.8B/14B</td></tr>
            <tr><td>Quantization</td><td>precision + compression</td><td>q4, q8, fp16</td></tr>
            <tr><td>Architecture</td><td>training and specialization</td><td>Mistral, CodeLlama, DeepSeek Coder</td></tr>
            <tr><td>Hardware</td><td>deployment practicality</td><td>CPU-only vs {hardware_name}</td></tr>
            <tr><td>Workload</td><td>request shape</td><td>chat, summary, code, search, embeddings</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <div class="section-title blue">Measurement Pipeline</div>
        <div class="stack">
          <div class="stack-row"><div class="stack-tag tag-fixtures">Fixtures</div><div class="stack-desc">Deterministic prompts for code, docs, chat, labels, search documents, and embedding batches.</div></div>
          <div class="stack-row"><div class="stack-tag tag-client">Client</div><div class="stack-desc">Ollama / OpenAI-compatible adapter with row-level request logs and repeatable model registry.</div></div>
          <div class="stack-row"><div class="stack-tag tag-energy">Energy</div><div class="stack-desc">NVML samples GPU watts, utilization, and VRAM; CPU uses RAPL when exposed or a labeled utilization/TDP estimate.</div></div>
          <div class="stack-row"><div class="stack-tag tag-visual">Visuals</div><div class="stack-desc">pandas summaries generate scaling curves, frontiers, heatmaps, waterfalls, timelines, and poster-ready assets.</div></div>
        </div>
        <p class="note">For Ollama, tokenization is reported as prompt evaluation / prefill because the server does not expose a pure tokenizer-only timer.</p>
      </div>

      <div class="section">
        <div class="section-title">Models Tried</div>
        <div class="chips">{chips}</div>
        <p class="note">The same code supports additional Ollama, vLLM, Hugging Face, and API-compatible endpoints.</p>
      </div>

      <div class="section">
        <div class="section-title blue">Scaling Laws</div>
        <div class="figure">
          <div class="fig-caption"><span>Experiment A - same architecture, different sizes</span><span>quality + joules</span></div>
          <div class="fig-body tight"><img class="chart chart-xs" src="experiment_a_scaling_curves.png" alt="Scaling curves"></div>
        </div>
        <div class="figure">
          <div class="fig-caption"><span>Energy vs accuracy frontier</span><span>tokens-per-joule lens</span></div>
          <div class="fig-body tight"><img class="chart chart-xs" src="energy_accuracy_frontier.png" alt="Energy accuracy frontier"></div>
        </div>
      </div>
    </div>

    <div class="col">
      <div class="section">
        <div class="section-title">What Happens During One LLM Request?</div>
        <p>Each request moves through CPU preparation, serving overhead, GPU inference, KV-cache growth, sampling, and response parsing. The benchmark records row-level traces instead of one opaque latency number.</p>
        <div class="figure">
          <div class="fig-caption"><span>Request lifecycle observability view</span><span>hardware + latency + energy</span></div>
          <div class="fig-body tight"><img class="chart chart-md" src="request_lifecycle.svg" alt="Request lifecycle"></div>
        </div>
      </div>

      <div class="section">
        <div class="section-title blue">Quantization Efficiency</div>
        <p>Precision is a deployment lever: it changes VRAM pressure, memory traffic, latency, and energy before application code changes.</p>
        <div class="figure">
          <div class="fig-caption"><span>Experiment B - same model, different quantization</span><span>VRAM + energy + tokens/J</span></div>
          <div class="fig-body tight"><img class="chart chart-lg" src="experiment_b_quantization.png" alt="Quantization efficiency"></div>
        </div>
        <div class="callout"><strong>Field heuristic:</strong> 4-bit quantized models often retain useful quality at a fraction of the memory and active energy cost.</div>
        <div class="figure">
          <div class="fig-caption"><span>Quantization efficiency waterfall</span><span>q4 savings vs fp16</span></div>
          <div class="fig-body tight"><img class="chart chart-sm" src="experiment_b_efficiency_waterfall.png" alt="Quantization waterfall"></div>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Architecture &amp; Workload Specialization</div>
        <p>
          Similar-size models do not behave the same. Coding, chat, summarization, and semantic search stress different parts of the stack and different training priors.
        </p>
        <div class="figure">
          <div class="fig-caption"><span>Experiment C - same size, different architecture</span><span>workload matrix</span></div>
          <div class="fig-body tight"><img class="chart chart-md" src="experiment_c_workload_heatmap.png" alt="Architecture heatmap"></div>
        </div>
      </div>

    </div>

    <div class="col">
      <div class="section">
        <div class="section-title">CPU vs GPU Deployment</div>
        <p>
          CPU rows force Ollama with <code>num_gpu=0</code>; GPU rows let Ollama place the model on {hardware_name}. This isolates practical hardware deployment tradeoffs for the same benchmark code.
        </p>
        <div class="figure">
          <div class="fig-caption"><span>Experiment D - throughput</span><span>CPU-only vs GPU</span></div>
          <div class="fig-body tight"><img class="chart chart-sm" src="experiment_d_cpu_gpu.png" alt="CPU GPU throughput"></div>
        </div>
        <table>
          <thead><tr><th>Workload / Model</th><th>CPU</th><th>GPU</th><th>Takeaway</th></tr></thead>
          <tbody>{selected_cpu_gpu_rows()}</tbody>
        </table>
      </div>

      <div class="section">
        <div class="section-title blue">Latency &amp; Energy Flow</div>
        <div class="figure">
          <div class="fig-caption"><span>CPU vs GPU waterfall latency</span><span>network + prefill + inference + post</span></div>
          <div class="fig-body tight"><img class="chart chart-sm" src="experiment_d_latency_waterfall.png" alt="Latency waterfall"></div>
        </div>
        <div class="figure">
          <div class="fig-caption"><span>Full-system energy flow</span><span>request lifecycle</span></div>
          <div class="fig-body tight"><img class="chart chart-xs" src="experiment_e_energy_sankey.svg" alt="Energy flow"></div>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Energy Accounting</div>
        <table>
          <thead><tr><th>Metric</th><th>How It Is Calculated</th><th>Use</th></tr></thead>
          <tbody>
            <tr><td><code>gpu_energy_j</code></td><td>integrated NVML watts over request time</td><td>raw device draw</td></tr>
            <tr><td><code>net_gpu_energy_j</code></td><td>GPU joules minus idle baseline x elapsed</td><td>active inference view</td></tr>
            <tr><td><code>cpu_energy_j</code></td><td>RAPL if available, otherwise TDP x utilization x time</td><td>CPU rows</td></tr>
            <tr><td><code>tokens/J</code></td><td>tokens divided by active device joules</td><td class="best">best field metric</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <div class="section-title">Practical Heuristics</div>
        <table>
          <tbody>
            <tr><td>Benchmark workloads, not model cards.</td><td>Chat, code, RAG, and embeddings stress different bottlenecks.</td></tr>
            <tr><td>Quantization is infrastructure.</td><td>It reduces memory movement and often improves tokens per joule.</td></tr>
            <tr><td>GPUs win with concurrency.</td><td>Small one-off calls hide GPU benefits; batches reveal them.</td></tr>
            <tr><td>Keep two energy views.</td><td>Raw energy for facility cost, net active energy for model comparison.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="footer">
    <div class="refs">
      <strong>Artifacts:</strong> reproducible Python benchmark framework, configs, notebooks, CSV metrics, NVML power traces, generated PNG/SVG assets, and this printable poster.<br>
      <strong>Open repository:</strong> <a href="https://github.com/shivaylamba/pycon-2026-poster">github.com/shivaylamba/pycon-2026-poster</a>
      &nbsp;|&nbsp; <strong>Inspired by:</strong> Alizadeh et al., <em>Language Models in Software Development Tasks: An Experimental Analysis of Energy and Accuracy</em>, arXiv:2412.00329.
    </div>
    <div class="qr-slot">QR<br>repo</div>
  </div>
  <div class="conf-bar">
    <span>PyCon US Poster Session</span>
    <span>Python scripts + pandas + matplotlib + Plotly + Ollama + NVML</span>
  </div>
</div>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
