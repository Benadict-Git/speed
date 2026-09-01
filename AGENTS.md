# AGENTS.md

Concise OpenCode-specific guidance for this repo. `CLAUDE.md` already covers
architecture, the two-fork layout, dataset/weights sourcing, and command
examples — read it first. This file adds only setup, path, and workflow
gotchas that hit on the first session.

## Environment

- Use `.venv` at the repo root. It's a `python3.11` venv (pyenv) with torch
  CPU-only and the heavy extras (`timm`, `mmcv`-like deps, `efficientnet_pytorch`).
- `ultralytics` is **not** pip-installed in `.venv`. The two forks each vendor
  their own `ultralytics/` package — pick one per task and prepend it to
  `PYTHONPATH` (or `pip install -e` the fork you need, then uninstall before
  switching):
  ```
  PYTHONPATH=ultralytics-distill .venv/bin/python ...
  PYTHONPATH=ultralytics-yolo11 .venv/bin/python ...
  ```
- Don't `pip install ultralytics` from PyPI on top of this — it shadows the
  vendored fork the same way and breaks SBP-YOLO's custom modules.
- CUDA is **not** available in `.venv` (torch is CPU build). Training/val
  scripts (`val.py` hardcodes `device="0"`, `distill.py` hardcodes
  `device='0'`) will fail until run on a CUDA host.

## Which fork to use

- `ultralytics-yolo11/` — for training (`train.py`), validation (`val.py`),
  and FPS measurement (`get_FPS.py`). Architecture configs live in
  `models/SBP-YOLO.yaml` and `models/SBP-YOLO-s.yaml`.
- `ultralytics-distill/` — for knowledge distillation (`distill.py`) **and**
  for loading the official `pre_trained_weights/sbp-yolo*.pt` checkpoints at
  inference. `detect.ipynb` and the working samples in `runs/detect/` were
  produced with this fork. Its `runs/distill/SBP-YOLO/{best,last}.pt` is a
  working substitute if the external weights haven't been downloaded.
- Switching forks mid-session = import errors for missing modules
  (`Detect_Efficient`, `VoVGSCSPC`, `GhostConv`, etc.). Restart Python.

## Hardcoded paths to edit before running

Several scripts have absolute machine paths baked in. Update before invoking:

- `ultralytics-yolo11/val.py:3` — `model=` path (default `pre_trained_weights/sbp-yolo-nwdloss.pt`) and `data=` path.
- `ultralytics-yolo11/train.py:11` — `data_path` default
  (`datasets/yolo_0422-ALL_7K5_7_1_1.yaml`) does not exist; the dataset must
  be downloaded from the README Baidu/Google links.
- `ultralytics-distill/distill.py:10-30` — every `model`/`data`/
  `teacher_weights`/`teacher_cfg` path is `/home/server/Work/yolo11/mask/...`
  from the original authors' machine and will not resolve here.

## Inference quick-start

```python
# from repo root
import sys; sys.path.insert(0, "ultralytics-distill")
from ultralytics import YOLO
model = YOLO("pre_trained_weights/sbp-yolo.pt")
results = model.predict(source="img/", save=True, conf=0.25,
                        project="runs/detect", name="sbp_predictions",
                        exist_ok=True)
```

`detect.ipynb` is a working end-to-end version of this. It writes outputs to
`runs/detect/notebook/` (untracked).

## Things that don't exist in this repo

- No `tests/`, no `pytest`/`ruff`/lint config at the repo root, no
  `.github/`, no pre-commit hooks. The `ultralytics-yolo11/pyproject.toml`
  defines these for the vendored library but nothing invokes them here.
- No git-tracked `datasets/` folder. Both training and val yams reference
  it; download from the README links.
- The `pre_trained_weights/` directory IS committed (see `git ls-files`), but
  only some files; the canonical SBP-YOLO checkpoints are linked externally.

## Gotchas

- `ultralytics-yolo11/zzz/` is the default `project=` for `train.py` and has
  30+ experimental run folders. Don't `rm -rf` without checking with the
  user — ask first.
- `run/detect/` outputs (`sbp_predictions/`, `sbp_official_predictions/`,
  `notebook/`) are gitignored output dirs; reruns overwrite when `exist_ok=True`.
- `simplify=True` in `train.py:35` requires `onnx`/`onnxsim` to be installed;
  failures here are usually a missing dep, not a code bug.
- The two forks can silently diverge — when debugging a model-shape mismatch,
  check that you're loading the checkpoint with the fork that defines the
  matching `Detect_Efficient` / `VoVGSCSPC` modules.
