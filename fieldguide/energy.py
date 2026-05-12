from __future__ import annotations

import glob
import json
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EnergySummary:
    elapsed_s: float
    cpu_energy_j: Optional[float]
    gpu_energy_j: Optional[float]
    total_energy_j: Optional[float]
    cpu_util_avg_pct: Optional[float]
    gpu_power_avg_w: Optional[float]
    gpu_power_peak_w: Optional[float]
    gpu_util_avg_pct: Optional[float]
    gpu_mem_peak_mb: Optional[float]
    gpu_mem_avg_mb: Optional[float]
    gpu_count: int
    source: str

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class EnergyTrace:
    """Samples CPU utilization, RAPL if available, and NVIDIA GPU power via NVML."""

    def __init__(self, interval_s: float = 0.1, gpu_indices: Optional[List[int]] = None, cpu_tdp_watts: Optional[float] = None) -> None:
        self.interval_s = interval_s
        self.gpu_indices = gpu_indices
        self.cpu_tdp_watts = cpu_tdp_watts
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = 0.0
        self._ended = 0.0
        self._rapl_start: Optional[float] = None
        self._rapl_end: Optional[float] = None
        self._psutil = None
        self._nvml = None
        self._gpu_handles: List[Any] = []
        self.samples: List[Dict[str, float]] = []

    def __enter__(self) -> "EnergyTrace":
        self._setup_psutil()
        self._setup_nvml()
        self._rapl_start = read_rapl_joules()
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval_s * 5))
        self._ended = time.perf_counter()
        self._rapl_end = read_rapl_joules()
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _setup_psutil(self) -> None:
        try:
            import psutil
        except Exception:
            self._psutil = None
            return
        self._psutil = psutil
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def _setup_nvml(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            indices = self.gpu_indices if self.gpu_indices is not None else list(range(count))
            self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices]
            self._nvml = pynvml
        except Exception:
            self._gpu_handles = []
            self._nvml = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.perf_counter()
            cpu_pct = 0.0
            if self._psutil:
                try:
                    cpu_pct = float(self._psutil.cpu_percent(interval=None))
                except Exception:
                    cpu_pct = 0.0
            gpu_power = 0.0
            gpu_util = 0.0
            gpu_mem_mb = 0.0
            gpu_seen = 0
            if self._nvml:
                for handle in self._gpu_handles:
                    try:
                        gpu_power += self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_util += float(util.gpu)
                        gpu_mem_mb += float(mem.used) / (1024**2)
                        gpu_seen += 1
                    except Exception:
                        pass
            self.samples.append(
                {
                    "t_rel_s": now - self._started if self._started else 0.0,
                    "cpu_util_pct": cpu_pct,
                    "gpu_power_w": gpu_power,
                    "gpu_util_pct": gpu_util / gpu_seen if gpu_seen else 0.0,
                    "gpu_mem_mb": gpu_mem_mb,
                }
            )
            self._stop.wait(self.interval_s)

    def summary(self) -> EnergySummary:
        elapsed = max(0.0, self._ended - self._started)
        cpu_values = [s["cpu_util_pct"] for s in self.samples]
        gpu_powers = [s["gpu_power_w"] for s in self.samples]
        gpu_utils = [s["gpu_util_pct"] for s in self.samples]
        gpu_mems = [s["gpu_mem_mb"] for s in self.samples]
        cpu_avg = mean(cpu_values)
        gpu_energy = integrate(self.samples, "gpu_power_w")
        cpu_energy = self._cpu_energy(elapsed, cpu_avg)
        values = [value for value in (cpu_energy, gpu_energy) if value is not None]
        source_parts = []
        if self._rapl_start is not None and self._rapl_end is not None:
            source_parts.append("cpu:rapl")
        elif self.cpu_tdp_watts:
            source_parts.append("cpu:tdp_estimate")
        else:
            source_parts.append("cpu:unavailable")
        source_parts.append("gpu:nvml" if gpu_powers else "gpu:unavailable")
        return EnergySummary(
            elapsed_s=elapsed,
            cpu_energy_j=cpu_energy,
            gpu_energy_j=gpu_energy,
            total_energy_j=sum(values) if values else None,
            cpu_util_avg_pct=cpu_avg,
            gpu_power_avg_w=mean(gpu_powers),
            gpu_power_peak_w=max(gpu_powers) if gpu_powers else None,
            gpu_util_avg_pct=mean(gpu_utils),
            gpu_mem_peak_mb=max(gpu_mems) if gpu_mems else None,
            gpu_mem_avg_mb=mean(gpu_mems),
            gpu_count=len(self._gpu_handles),
            source=";".join(source_parts),
        )

    def _cpu_energy(self, elapsed: float, cpu_avg_pct: Optional[float]) -> Optional[float]:
        if self._rapl_start is not None and self._rapl_end is not None:
            delta = self._rapl_end - self._rapl_start
            if delta >= 0:
                return delta
        if self.cpu_tdp_watts and cpu_avg_pct is not None:
            return elapsed * self.cpu_tdp_watts * (cpu_avg_pct / 100.0)
        return None

    def write_samples(self, path: Path, run_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for sample in self.samples:
                row = {"run_id": run_id, **sample}
                handle.write(json.dumps(row) + "\n")


def integrate(samples: List[Dict[str, float]], key: str) -> Optional[float]:
    if not samples:
        return None
    if len(samples) == 1:
        return samples[0][key]
    energy = 0.0
    for a, b in zip(samples, samples[1:]):
        energy += ((a[key] + b[key]) / 2.0) * max(0.0, b["t_rel_s"] - a["t_rel_s"])
    return energy


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def read_rapl_joules() -> Optional[float]:
    paths = [Path(path) for path in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")]
    total = 0
    seen = 0
    for path in paths:
        try:
            if path.parent.parent.name != "powercap":
                continue
            total += int(path.read_text(encoding="utf-8").strip())
            seen += 1
        except Exception:
            pass
    return total / 1_000_000.0 if seen else None


def hardware_snapshot() -> Dict[str, Any]:
    info: Dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(), "processor": platform.processor()}
    try:
        import psutil
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        gpus = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
            gpus.append({"index": index, "name": name, "memory_gb": round(mem, 2)})
        pynvml.nvmlShutdown()
        info["gpus"] = gpus
    except Exception:
        info["gpus"] = []
    return info
