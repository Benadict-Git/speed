# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SBP-YOLO is a lightweight YOLO11-based object detector specialized for real-time
detection of road **speed bumps** and **potholes**, targeting embedded vehicle
suspension control systems. Paper: https://arxiv.org/abs/2508.01339

Key architectural changes vs. stock YOLO11 (see `ultralytics-yolo11/models/SBP-YOLO.yaml`
and `SBP-YOLO-s.yaml`):
- GhostConv + VoVGSCSPC in the backbone/neck for cheaper multi-scale feature extraction
- An added P2 detection branch for small-object detection
- A Lightweight Efficient Detection Head (LEDH)
- Training uses NWD loss, BCKD knowledge distillation, and Albumentations augmentation

## Repository layout

This is not a single package — it contains **two separate forks of Ultralytics**,
each a full vendored copy of the `ultralytics` library with local modifications:

- `ultralytics-yolo11/` — the primary fork used for training/val (`train.py`, `val.py`).
  Has its own `pyproject.toml` (installable, `pip install -e .`) and the
  `models/SBP-YOLO.yaml` / `models/SBP-YOLO-s.yaml` architecture configs.
- `ultralytics-distill/` — a second, heavier fork used for **knowledge distillation**
  (`distill.py`, using `ultralytics.models.yolo.detect.distill.DetectionDistiller`).
  Its `ultralytics/nn/extra_modules/` and `ultralytics/nn/backbone/` pull in many
  optional/experimental backbones and CUDA ops (DCNv4, mamba/selective_scan,
  TransNeXt, rational_kat_cu) that are **not required** for SBP-YOLO inference —
  only import what a given checkpoint actually needs.
  - `ultralytics-distill/runs/distill/SBP-YOLO/weights/{best,last}.pt` is a real
    trained checkpoint committed in this repo, with training artifacts
    (PR/F1/confusion-matrix curves, `results.csv`, `args.yaml`) alongside it —
    useful as a working weights file when the released `pre_trained_weights/*.pt`
    from the README's Google Drive links haven't been downloaded locally.

Because each fork vendors its own `ultralytics/` package, **do not `pip install`
the two forks into the same environment** — pick one fork's `ultralytics/` (via a
local editable install or `PYTHONPATH`) per task, matching whichever fork produced
the checkpoint/config you're working with.

Top-level `img/` holds ad hoc sample road images used for manual inference checks;
it is not part of the training dataset pipeline.

## Commands

There is no single unified CLI; everything is invoked as plain Python scripts from
inside the relevant fork directory.

Training (from `ultralytics-yolo11/`):
```
python train.py --cfg models/SBP-YOLO.yaml --data datasets/<your_dataset>.yaml
```
`train.py` hardcodes `models/SBP-YOLO.yaml` and a dataset yaml
(`datasets/yolo_0422-ALL_7K5_7_1_1.yaml`) as defaults — that dataset is not
included in this repo and must be supplied (see Dataset links in `README.md`).

Validation (from `ultralytics-yolo11/`, edit the hardcoded `model=`/`data=` paths
in `val.py` first):
```
python val.py
```

Knowledge distillation (from `ultralytics-distill/`, edit the hardcoded
`param_dict` paths in `distill.py` first — `model`/`data`/`teacher_weights`/
`teacher_cfg` are absolute paths from the original authors' machine):
```
python distill.py
```

Inference on new images (either fork's `ultralytics.YOLO` API):
```python
from ultralytics import YOLO
model = YOLO("path/to/weights.pt")
results = model.predict(source="img/", save=True, conf=0.25)
```

## Weights

Released pre-trained weights (`sbp-yolo.pt`, `sbp-yolo-nwdloss.pt`,
`yolo11-ledh-ghostconv.pt`, `yolo11-p2-ledh.pt`, `yolo11-p2.pt`, `yolo11.pt`) are
distributed externally via Baidu Netdisk / Google Drive links in `README.md`, not
committed to git. `val.py` expects them under a local `pre_trained_weights/`
directory that does not exist until you download them. The dataset used for
training/eval is similarly external-only (Baidu/Google Drive links in `README.md`).
