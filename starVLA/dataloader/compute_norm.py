"""Compute and export the exact statistics used by the StarVLA transforms.

Example::

    PYTHONPATH=. python -m starVLA.dataloader.compute_norm \
        --config_yaml examples/realRobots/EgoS2/train_files/starvla_qwenOFT_EgoS2_Adamu.yaml \
        --data_root_dir /path/to/egos2_starvla_parent \
        --output_dir /path/to/egos2_norm

The command constructs the same LeRobot dataset as training, disables random
augmentation, refreshes stale ``meta/stats_gr00t.json`` caches when parquet
files changed, and writes the merged ``dataset_statistics.json`` consumed by
checkpoint loading.  It does not load a model or allocate a CUDA device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.share_tools import apply_config_compat


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def compute_norm(
    config_yaml: str | Path,
    *,
    data_root_dir: str | Path | None = None,
    data_mix: str | None = None,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Compute statistics and return the output ``dataset_statistics.json``."""

    cfg = OmegaConf.load(str(config_yaml))
    apply_config_compat(cfg)
    data_cfg = cfg.datasets.vla_data

    if data_root_dir is not None:
        data_cfg.data_root_dir = str(data_root_dir)
    if data_mix is not None:
        data_cfg.data_mix = data_mix

    # Norms describe the source dataset, not one random augmented view.
    if "mirror_augmentation" in data_cfg:
        data_cfg.mirror_augmentation.enabled = False

    root = Path(data_cfg.data_root_dir)
    output = Path(output_dir) if output_dir is not None else root / "norm"
    output.mkdir(parents=True, exist_ok=True)

    if force:
        from starVLA.dataloader.gr00t_lerobot.datasets import LE_ROBOT_STATS_FILENAME
        from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

        for dataset_name, _, _ in DATASET_NAMED_MIXTURES[str(data_cfg.data_mix)]:
            cache_path = root / dataset_name / LE_ROBOT_STATS_FILENAME
            if cache_path.exists():
                cache_path.unlink()

    dataset = get_vla_dataset(data_cfg=data_cfg, mode="eval")
    statistics_path = output / "dataset_statistics.json"
    dataset.save_dataset_statistics(statistics_path)

    summary = {
        "config_yaml": str(Path(config_yaml).resolve()),
        "data_root_dir": str(root.resolve()),
        "data_mix": str(data_cfg.data_mix),
        "robot_type": str(data_cfg.robot_type),
        "statistics_path": str(statistics_path.resolve()),
        "action_statistics_policy": (
            "source action rows only; synthetic positions beyond an episode "
            "boundary are excluded from horizon-based delta/rel statistics"
        ),
        "datasets": [
            {
                "dataset_name": item.dataset_name,
                "dataset_path": str(item.dataset_path.resolve()),
                "num_transitions": int(len(item)),
                "num_trajectories": int(len(item.trajectory_ids)),
                "metadata": _jsonable(item.metadata),
            }
            for item in dataset.datasets
        ],
        "merged_statistics": _jsonable(dataset.merged_metadata),
    }
    summary_path = output / "norm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Computed dataset norm: {statistics_path}")
    print(f"Wrote norm summary: {summary_path}")
    return statistics_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--data_root_dir", default=None)
    parser.add_argument("--data_mix", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild the targeted stats cache")
    args = parser.parse_args()
    compute_norm(
        args.config_yaml,
        data_root_dir=args.data_root_dir,
        data_mix=args.data_mix,
        output_dir=args.output_dir,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
