from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


LATENCY_COMPONENTS = [
    ("network_s", "Network"),
    ("tokenization_s", "Tokenization / prefill"),
    ("inference_s", "Inference"),
    ("postprocess_s", "Post-processing"),
]


def make_all_plots(metrics_path: str | Path, out_dir: str | Path) -> List[Path]:
    metrics_path = Path(metrics_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_path)
    df = add_energy_views(df)
    enriched_path = out_path / "metrics_with_energy_views.csv"
    df.to_csv(enriched_path, index=False)
    summary = summarize(df)
    summary_path = out_path / "summary_by_workload_model.csv"
    summary.to_csv(summary_path, index=False)
    created = [enriched_path, summary_path]
    for workload in sorted(summary["workload"].dropna().unique()):
        subset = summary[summary["workload"] == workload].copy()
        if subset.empty:
            continue
        created.append(plot_latency_breakdown(subset, out_path / f"latency_waterfall_{workload}.png"))
        created.append(plot_cost_bars(subset, out_path / f"cost_bars_{workload}.png"))
        created.append(plot_energy_bars(subset, out_path / f"energy_cpu_gpu_{workload}.png"))
        created.append(plot_active_device_energy_bars(subset, out_path / f"energy_active_device_{workload}.png"))
    return created


def add_energy_views(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hardware = df.get("hardware_profile", pd.Series("", index=df.index)).astype(str).str.lower()
    cpu_energy = df.get("cpu_energy_j", pd.Series(0.0, index=df.index)).fillna(0.0)
    gpu_energy = df.get("gpu_energy_j", pd.Series(0.0, index=df.index)).fillna(0.0)
    elapsed = df.get("elapsed_s", pd.Series(0.0, index=df.index)).fillna(0.0)
    gpu_power = df.get("gpu_power_avg_w", pd.Series(float("nan"), index=df.index))
    idle_candidates = gpu_power[hardware.eq("cpu")].dropna()
    gpu_idle_power_w = float(idle_candidates.median()) if not idle_candidates.empty else 0.0
    gpu_net_energy = (gpu_energy - gpu_idle_power_w * elapsed).clip(lower=0.0)
    df["gpu_idle_power_w_used"] = gpu_idle_power_w
    df["net_gpu_energy_j"] = gpu_net_energy
    df["active_device_energy_j"] = cpu_energy.where(hardware.eq("cpu"), gpu_net_energy)
    tokens = df.get("total_tokens", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["active_device_energy_j_per_1k_tokens"] = (
        df["active_device_energy_j"] / tokens.where(tokens > 0) * 1000.0
    )
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "request_wall_s",
        "network_s",
        "tokenization_s",
        "inference_s",
        "postprocess_s",
        "total_latency_s",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cpu_energy_j",
        "gpu_energy_j",
        "gpu_idle_power_w_used",
        "net_gpu_energy_j",
        "total_energy_j",
        "api_cost_usd",
        "energy_cost_usd",
        "hardware_cost_usd",
        "total_cost_usd",
        "cost_per_1k_tokens_usd",
        "energy_j_per_1k_tokens",
        "active_device_energy_j",
        "active_device_energy_j_per_1k_tokens",
    ]
    existing_numeric = [column for column in numeric if column in df.columns]
    group_cols = ["workload", "model_label", "model", "hardware_profile", "hardware_label"]
    summary = df.groupby(group_cols, dropna=False)[existing_numeric].mean().reset_index()
    if "is_correct" in df.columns:
        accuracy = (
            df.dropna(subset=["is_correct"])
            .groupby(group_cols, dropna=False)["is_correct"]
            .mean()
            .reset_index(name="accuracy")
        )
        summary = summary.merge(accuracy, on=group_cols, how="left")
    summary["series_label"] = summary["model_label"].astype(str) + " | " + summary["hardware_label"].astype(str)
    return summary


def plot_latency_breakdown(df: pd.DataFrame, output_path: Path) -> Path:
    subset = df.sort_values("total_latency_s", ascending=True).tail(12)
    labels = subset["series_label"].tolist()
    left = [0.0] * len(subset)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(subset))))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for (column, label), color in zip(LATENCY_COMPONENTS, colors):
        values = subset[column].fillna(0.0).tolist()
        ax.barh(labels, values, left=left, label=label, color=color)
        left = [l + v for l, v in zip(left, values)]
    ax.set_xlabel("Mean latency (seconds)")
    ax.set_title(f"Latency Waterfall: {subset['workload'].iloc[0]}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_cost_bars(df: pd.DataFrame, output_path: Path) -> Path:
    subset = df.sort_values("total_cost_usd", ascending=True).tail(12)
    labels = subset["series_label"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.42 * len(subset))), sharey=True)
    axes[0].barh(labels, subset["total_cost_usd"].fillna(0.0), color="#4C78A8")
    axes[0].set_title("Cost per request")
    axes[0].set_xlabel("USD")
    axes[1].barh(labels, subset["cost_per_1k_tokens_usd"].fillna(0.0), color="#E45756")
    axes[1].set_title("Cost per 1,000 tokens")
    axes[1].set_xlabel("USD")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(f"Cost Comparison: {subset['workload'].iloc[0]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_energy_bars(df: pd.DataFrame, output_path: Path) -> Path:
    subset = df.sort_values("total_energy_j", ascending=True).tail(12)
    labels = subset["series_label"].tolist()
    cpu = subset["cpu_energy_j"].fillna(0.0).tolist()
    gpu = subset["gpu_energy_j"].fillna(0.0).tolist()
    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(subset))))
    ax.barh(labels, cpu, color="#72B7B2", label="CPU")
    ax.barh(labels, gpu, left=cpu, color="#F58518", label="GPU")
    ax.set_xlabel("Mean energy (joules)")
    ax.set_title(f"CPU vs GPU Energy: {subset['workload'].iloc[0]}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_active_device_energy_bars(df: pd.DataFrame, output_path: Path) -> Path:
    subset = df.sort_values("active_device_energy_j", ascending=True).tail(12)
    labels = subset["series_label"].tolist()
    colors = subset["hardware_profile"].map({"cpu": "#72B7B2", "gpu": "#F58518"}).fillna("#4C78A8")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(subset))))
    ax.barh(labels, subset["active_device_energy_j"].fillna(0.0), color=colors)
    ax.set_xlabel("Mean net active-device energy (joules)")
    ax.set_title(f"Net Active Device Energy: {subset['workload'].iloc[0]}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
