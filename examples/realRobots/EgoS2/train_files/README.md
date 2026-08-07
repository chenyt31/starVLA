# StarVLA × EgoS2 Adamu

This integration uses the registry names `EgoS2_Adamu` and
`EgoS2_Adamu_Open`. The dataset state and action are both 36-D:

```text
state : [left absolute EEF pose (7), left absolute hand pose (11),
         right absolute EEF pose (7), right absolute hand pose (11)]
action: [left delta EEF pose (7), left absolute hand pose (11),
         right delta EEF pose (7), right absolute hand pose (11)]
```

The EEF pose is expressed in the camera frame. State is evaluated at the
current frame; EEF action is the motion from the current frame to the next
frame, while hand action targets the next hand pose.

The generated dataset must contain `meta/modality.json` and
`meta/stats_gr00t.json`. Run the dataloader smoke test from the StarVLA
checkout:

```bash
PYTHONPATH=. python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/realRobots/EgoS2/train_files/starvla_qwenOFT_EgoS2_Adamu.yaml \
  --data_root_dir /path/to/egos2_starvla_parent
```

The delta EEF quaternion remains `absolute: false`, but uses the explicit
`padding: first_last` extension in `meta/modality.json`. This repeats the
terminal identity quaternion when an 8-step action window crosses the end of
an episode; zero-padding a quaternion would produce an invalid rotation.

For training, first replace the VLM path in the YAML (or override it on the
command line), then launch `starVLA/training/train_starvla.py` with the same
config. The action head is newly sized for this EgoS2 embodiment.

## Norm, spike and overfit diagnostics

Compute the exact norm used by the dataloader before a run:

```bash
PYTHONPATH=. python -m starVLA.dataloader.compute_norm \
  --config_yaml examples/realRobots/EgoS2/train_files/starvla_qwenOFT_EgoS2_Adamu.yaml \
  --data_root_dir /path/to/egos2_starvla_parent \
  --output_dir /path/to/egos2_norm
```

Training writes `dataset_statistics.json` into the run directory.  The
`loss_spikes_rank*.jsonl` files contain the numeric dataloader index and the
physical `dataset/episode/step` ID for each detected spike.

To replay a small deterministic subset and inspect whether the action head can
memorize it, enable both overfit switches in the YAML or with dotlist
overrides:

```bash
PYTHONPATH=. python starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/EgoS2/train_files/starvla_qwenOFT_EgoS2_Adamu.yaml \
  --datasets.vla_data.overfit.enabled=true \
  --datasets.vla_data.overfit.num_samples=8 \
  --trainer.overfit.enabled=true \
  --trainer.max_train_steps=200
```

The run then contains `overfit_trace_rank*.jsonl` and
`overfit_report_rank*.json`. The default augmentation is train-only and
mirrors 50% of examples: image width flip, left/right action exchange,
camera-frame reflection (`[-x, y, z]`, `[w, x, -y, -z]`) and the matching
Adamu state reflection.

To overfit the same fixed physical samples while also testing mirror
augmentation, add
`datasets.vla_data.overfit.apply_train_augmentations=true`. The loader keeps
overfit sampling deterministic but lets the mirror coin vary by epoch.

WanPI is available through
`train_files/starvla_wanPI_EgoS2_Adamu.yaml`. It synchronizes its action DiT
depth to Wan's transformer depth and defaults to a frozen Wan backbone for a
low-memory first regression; set `framework.freeze_world_model=false` only
after the projector/action-head path is stable.
