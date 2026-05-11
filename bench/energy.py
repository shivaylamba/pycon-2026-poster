from __future__ import annotations

import glob
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EnergyResult:
    elapsed_s: float
    cpu_energy_j: Optional[float]
    gpu_energy_j: Optional[float]
    cpu_util_avg_pct: Optional[float]
    gpu_power_avg_w: Optional[float]
    gpu_power_peak_w: Optional[float]
    gpu_count: int
    source: str

    @property
    def total_energy_j(self) -> Optional[float]:
        values = [value for value in [self.cpu_energy_j, self.gpu_energy_j] if value is not None]
        if not values:
            return None
        return float(sum(values))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "elapsed_s": self.elapsed_s,
            "cpu_energy_j": self.cpu_energy_j,
            "gpu_energy_j": self.gpu_energy_j,
            "total_energy_j": self.total_energy_j,
            "cpu_util_avg_pct": self.cpu_util_avg_pct,
            "gpu_power_avg_w": self.gpu_power_avg_w,
            "gpu_power_peak_w": self.gpu_power_peak_w,
            "gpu_count": self.gpu_count,
            "energy_source": self.source,
        }


class EnergySampler:
    """Measures GPU energy with NVML and CPU energy with RAPL when available.

    If RAPL is unavailable, set ``cpu_tdp_watts`` in the hardware profile to
    record a coarse utilization-based CPU energy estimate. The CSV columns make
    that source explicit so poster figures can distinguish measured vs estimated.
    """

    def __init__(
        self,
        *,
        interval_s: float = 0.1,
        gpu_indices: Optional[List[int]] = None,
        cpu_tdp_watts: Optional[float] = None,
    ) -> None:
        self.interval_s = interval_s
        self.gpu_indices = gpu_indices
        self.cpu_tdp_watts = cpu_tdp_watts
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._gpu_samples: List[tuple[float, float]] = []
        self._cpu_samples: List[float] = []
        self._rapl_start: Optional[float] = None
        self._rapl_end: Optional[float] = None
        self._started = 0.0
        self._ended = 0.0
        self._nvml = None
        self._gpu_handles: List[Any] = []
        self._psutil = None

    def __enter__(self) -> "EnergySampler":
        self._started = time.perf_counter()
        self._rapl_start = read_rapl_joules()
        self._setup_psutil()
        self._setup_nvml()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s * 4)
        self._ended = time.perf_counter()
        self._rapl_end = read_rapl_joules()
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def result(self) -> EnergyResult:
        elapsed = max(0.0, self._ended - self._started)
        cpu_avg = _mean(self._cpu_samples)
        cpu_energy = self._cpu_energy(elapsed, cpu_avg)
        gpu_energy = self._gpu_energy()
        gpu_powers = [power for _, power in self._gpu_samples]
        source_parts = []
        if self._rapl_start is not None and self._rapl_end is not None:
            source_parts.append("cpu:rapl")
        elif self.cpu_tdp_watts is not None:
            source_parts.append("cpu:tdp_estimate")
        else:
            source_parts.append("cpu:unavailable")
        source_parts.append("gpu:nvml" if self._gpu_samples else "gpu:unavailable")
        return EnergyResult(
            elapsed_s=elapsed,
            cpu_energy_j=cpu_energy,
            gpu_energy_j=gpu_energy,
            cpu_util_avg_pct=cpu_avg,
            gpu_power_avg_w=_mean(gpu_powers),
            gpu_power_peak_w=max(gpu_powers) if gpu_powers else None,
            gpu_count=len(self._gpu_handles),
            source=";".join(source_parts),
        )

    def _setup_psutil(self) -> None:
        try:
            import psutil
        except ImportError:
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
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            indices = self.gpu_indices if self.gpu_indices is not None else list(range(count))
            self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in indices]
            self._nvml = pynvml
        except Exception:
            self._gpu_handles = []
            self._nvml = None

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            now = time.perf_counter()
            if self._psutil:
                try:
                    self._cpu_samples.append(float(self._psutil.cpu_percent(interval=None)))
                except Exception:
                    pass
            if self._nvml and self._gpu_handles:
                total_power_w = 0.0
                for handle in self._gpu_handles:
                    try:
                        total_power_w += self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except Exception:
                        pass
                self._gpu_samples.append((now, total_power_w))
            self._stop.wait(self.interval_s)

    def _cpu_energy(self, elapsed: float, cpu_avg: Optional[float]) -> Optional[float]:
        if self._rapl_start is not None and self._rapl_end is not None:
            delta = self._rapl_end - self._rapl_start
            if delta >= 0:
                return delta
        if self.cpu_tdp_watts is not None and cpu_avg is not None:
            return elapsed * self.cpu_tdp_watts * (cpu_avg / 100.0)
        return None

    def _gpu_energy(self) -> Optional[float]:
        if not self._gpu_samples:
            return None
        if len(self._gpu_samples) == 1:
            return self._gpu_samples[0][1] * self.interval_s
        energy_j = 0.0
        for (t0, p0), (t1, p1) in zip(self._gpu_samples, self._gpu_samples[1:]):
            energy_j += ((p0 + p1) / 2.0) * max(0.0, t1 - t0)
        return energy_j


def read_rapl_joules() -> Optional[float]:
    paths = [Path(path) for path in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")]
    total_uj = 0
    seen = 0
    for path in paths:
        try:
            # Keep top-level package domains and avoid nested DRAM/uncore zones.
            if path.parent.parent.name != "powercap":
                continue
            total_uj += int(path.read_text(encoding="utf-8").strip())
            seen += 1
        except Exception:
            continue
    if not seen:
        return None
    return total_uj / 1_000_000.0


def hardware_snapshot() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
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
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
            gpus.append({"index": index, "name": name, "memory_gb": round(memory, 2)})
        info["gpus"] = gpus
        pynvml.nvmlShutdown()
    except Exception:
        info["gpus"] = []
    return info


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))

