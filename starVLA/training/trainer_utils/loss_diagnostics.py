"""Local diagnostics for unstable and intentionally overfit training runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return value


def _sample_metadata(example: dict[str, Any], position: int) -> dict[str, Any]:
    """Extract JSON-safe identity fields without coupling to a dataset class."""

    raw_mask = example.get("action_valid_mask")
    if raw_mask is None:
        action_valid_mask = [1]
    else:
        action_valid_mask = np.asarray(raw_mask, dtype=np.float32).reshape(-1).tolist()
        if not action_valid_mask:
            action_valid_mask = [1]
    valid_fraction = float(np.mean(action_valid_mask))
    padding_count = int(sum(value <= 0.0 for value in action_valid_mask))

    result = {
        "batch_position": int(position),
        "sample_index": _scalar(example.get("sample_index")),
        "sample_id": str(example.get("sample_id", f"batch_position_{position}")),
        "dataset_name": str(example.get("dataset_name", "unknown")),
        "trajectory_id": _scalar(example.get("trajectory_id")),
        "base_index": _scalar(example.get("base_index")),
        "is_mirrored": bool(example.get("is_mirrored", False)),
        "action_valid_mask": [int(value > 0.0) for value in action_valid_mask],
        "action_valid_fraction": valid_fraction,
        "action_padding_count": padding_count,
    }
    return result


def _as_losses(losses: Any) -> np.ndarray | None:
    if losses is None:
        return None
    if isinstance(losses, torch.Tensor):
        values = losses.detach().float().cpu().reshape(-1).numpy()
    else:
        values = np.asarray(losses, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None
    return values


class LossSpikeTracker:
    """Write sample identities for unusually high per-sample losses.

    The reference is an EMA of the batch median.  This works with batch size 1
    (the common EgoS2 smoke configuration) while still being robust to one bad
    item in a larger batch.  Each distributed rank writes its own JSONL file,
    so no records are lost to concurrent append races.
    """

    def __init__(self, output_dir: str | Path, config: Any = None):
        self.config = config
        self.enabled = bool(_get(config, "enabled", True))
        self.warmup_steps = int(_get(config, "warmup_steps", 10))
        self.ratio_threshold = float(_get(config, "ratio_threshold", 4.0))
        absolute = _get(config, "absolute_threshold", None)
        self.absolute_threshold = None if absolute is None else float(absolute)
        self.ema_decay = float(_get(config, "ema_decay", 0.98))
        self.top_k = max(1, int(_get(config, "top_k", 8)))
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"loss_spike.ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.ratio_threshold <= 0:
            raise ValueError(f"loss_spike.ratio_threshold must be positive, got {self.ratio_threshold}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = _rank()
        self.path = self.output_dir / f"loss_spikes_rank{self.rank}.jsonl"
        # Always leave an explicit per-rank artifact, even when a run has no
        # spikes.  An empty file means the tracker was active and observed no
        # threshold crossings, rather than that diagnostics were not enabled.
        if self.enabled:
            self.path.touch(exist_ok=True)
        self._ema: float | None = None
        self._updates = 0

    @property
    def reference_loss(self) -> float | None:
        return self._ema

    def update(
        self,
        *,
        step: int,
        micro_step: int,
        batch: Iterable[dict[str, Any]],
        per_sample_losses: Any,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        values = _as_losses(per_sample_losses)
        if values is None:
            return {}
        examples = list(batch)
        if len(examples) != len(values):
            raise ValueError(
                f"Per-sample loss count {len(values)} does not match batch size {len(examples)}"
            )

        finite_values = values[np.isfinite(values)]
        batch_median = float(np.median(finite_values)) if finite_values.size else math.inf
        reference = self._ema
        threshold = None
        if reference is not None and math.isfinite(reference):
            threshold = reference * self.ratio_threshold
            if self.absolute_threshold is not None:
                threshold = max(threshold, self.absolute_threshold)

        records = []
        for position, (example, value) in enumerate(zip(examples, values)):
            is_nonfinite = not math.isfinite(float(value))
            is_spike = is_nonfinite or (
                threshold is not None
                and self._updates >= self.warmup_steps
                and float(value) >= threshold
            )
            if not is_spike:
                continue
            metadata = _sample_metadata(example, position)
            record = {
                "step": int(step),
                "micro_step": int(micro_step),
                "rank": int(self.rank),
                "loss": float(value) if math.isfinite(float(value)) else None,
                "reference_loss": None if reference is None else float(reference),
                "threshold": None if threshold is None else float(threshold),
                **metadata,
            }
            records.append(record)

        records.sort(key=lambda item: -math.inf if item["loss"] is None else item["loss"], reverse=True)
        records = records[: self.top_k]
        if records:
            with self.path.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

        if math.isfinite(batch_median):
            self._ema = (
                batch_median
                if self._ema is None
                else self.ema_decay * self._ema + (1.0 - self.ema_decay) * batch_median
            )
        self._updates += 1

        finite_for_max = values[np.isfinite(values)]
        valid_fractions = np.asarray(
            [float(_sample_metadata(example, position)["action_valid_fraction"])
             for position, example in enumerate(examples)],
            dtype=np.float64,
        )
        padding_counts = np.asarray(
            [int(_sample_metadata(example, position)["action_padding_count"])
             for position, example in enumerate(examples)],
            dtype=np.int64,
        )
        result: dict[str, Any] = {
            "loss_spike/count": len(records),
            "loss_spike/max_sample_loss": (
                float(np.max(finite_for_max)) if finite_for_max.size else float("inf")
            ),
            "action_valid_fraction/min": float(np.min(valid_fractions)),
            "action_padding_count/max": int(np.max(padding_counts)),
        }
        if records:
            result["loss_spike/sample_ids"] = ",".join(item["sample_id"] for item in records)
            result["loss_spike/sample_indices"] = ",".join(
                str(item["sample_index"]) for item in records
            )
        return result


class OverfitAnalyzer:
    """Record per-sample curves and emit a compact overfit report."""

    def __init__(self, output_dir: str | Path, config: Any = None):
        self.config = config
        self.enabled = bool(_get(config, "enabled", False))
        self.log_every = max(1, int(_get(config, "log_every", 1)))
        self.target_ratio = float(_get(config, "target_ratio", 0.1))
        target_loss = _get(config, "target_loss", None)
        self.target_loss = None if target_loss is None else float(target_loss)
        self.rank = _rank()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_dir / f"overfit_trace_rank{self.rank}.jsonl"
        self._updates = 0
        self._samples: dict[str, dict[str, Any]] = {}

    def update(
        self,
        *,
        step: int,
        micro_step: int,
        batch: Iterable[dict[str, Any]],
        per_sample_losses: Any,
    ) -> None:
        if not self.enabled:
            return
        values = _as_losses(per_sample_losses)
        if values is None:
            return
        examples = list(batch)
        if len(examples) != len(values):
            raise ValueError(
                f"Per-sample loss count {len(values)} does not match batch size {len(examples)}"
            )
        should_write = self._updates % self.log_every == 0
        records = []
        for position, (example, value) in enumerate(zip(examples, values)):
            metadata = _sample_metadata(example, position)
            sample_id = metadata["sample_id"]
            state = self._samples.setdefault(
                sample_id,
                {
                    **metadata,
                    "first_loss": float(value),
                    "first_step": int(step),
                    "min_loss": float(value),
                    "best_step": int(step),
                    "last_loss": float(value),
                    "last_step": int(step),
                    "observations": 0,
                },
            )
            value_float = float(value)
            state["min_loss"] = min(float(state["min_loss"]), value_float)
            if value_float <= float(state["min_loss"]):
                state["best_step"] = int(step)
            state["last_loss"] = value_float
            state["last_step"] = int(step)
            state["observations"] += 1
            if should_write:
                records.append(
                    {
                        "step": int(step),
                        "micro_step": int(micro_step),
                        "rank": int(self.rank),
                        "loss": value_float if math.isfinite(value_float) else None,
                        **metadata,
                    }
                )
        if records:
            with self.trace_path.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._updates += 1

    def finalize(self) -> Path | None:
        if not self.enabled:
            return None
        samples = []
        for state in self._samples.values():
            initial = float(state["first_loss"])
            final = float(state["last_loss"])
            ratio = final / max(abs(initial), 1e-12)
            success = ratio <= self.target_ratio
            if self.target_loss is not None:
                success = success or final <= self.target_loss
            samples.append({**state, "final_to_initial_ratio": ratio, "overfit_success": success})
        samples.sort(key=lambda item: item["sample_id"])
        successful = sum(bool(item["overfit_success"]) for item in samples)
        report = {
            "rank": int(self.rank),
            "updates": int(self._updates),
            "num_samples": len(samples),
            "successful_samples": int(successful),
            "success_fraction": successful / len(samples) if samples else 0.0,
            "target_ratio": self.target_ratio,
            "target_loss": self.target_loss,
            "samples": samples,
        }
        report_path = self.output_dir / f"overfit_report_rank{self.rank}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report_path
