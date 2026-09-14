# Antigravity 2.0 Prompt — Speed Bump Benchmark on Kaggle GPU

Copy everything below the horizontal rule into Antigravity 2.0.

---

## ROLE AND DIVISION OF LABOUR

You are building a reproducible comparative benchmark for an academic paper on vision-based speed bump detection. **You do not have a GPU.** All training and GPU measurement runs on Kaggle, driven through the Kaggle CLI (`kaggle kernels push`), in batch mode.

Your job splits cleanly:

- **You generate:** the dataset pipeline, one self-contained Kaggle notebook per run, the CLI orchestration scripts, the shared evaluation code, and all analysis, tables, and figures.
- **Kaggle executes:** training and GPU latency measurement.
- **You never:** run training locally, or assume a persistent machine holds state between runs.

The scientific validity of this study rests on holding the evaluation constant across models. Treat any shortcut that breaks comparability as a bug, not an optimization.

## KAGGLE PLATFORM CONSTRAINTS

These are hard limits. Design around them rather than discovering them at run time.

- Weekly GPU quota is approximately **30 hours**, floating (it varies with demand). Check the actual remaining quota before each batch and log it.
- Per-session cap is roughly **9 hours** for GPU sessions. Individual runs must finish well inside this. **Do not chain multiple training runs into one kernel.**
- **No persistent VM, no SSH, no terminal.** Every kernel starts from a clean image. Anything you need must be installed in-kernel or mounted from a Kaggle Dataset.
- Only `/kaggle/working` persists as kernel output, capped at 20 GB.
- Internet access must be explicitly enabled in kernel metadata and requires a phone-verified account. YOLOv13 needs it (installed from GitHub).
- Accelerator options: **P100 (16 GB)** or **T4 x2 (32 GB)**. Prefer T4 x2, but see the parallelism rule below.

## MODELS UNDER TEST

| ID | Checkpoint | Framework | NMS-free | Role in the study |
|---|---|---|---|---|
| `yolo26n` | `yolo26n.pt` | ultralytics | Yes (native) | Proposed model |
| `yolo11n` | `yolo11n.pt` | ultralytics | No | Predecessor generation; base of SBP-YOLO baseline |
| `yolov8n` | `yolov8n.pt` | ultralytics | No | Industry baseline anchor |
| `yolov10n` | `yolov10n.pt` | ultralytics | Yes | First NMS-free YOLO; isolates the NMS-free variable |
| `yolo12n` | `yolo12n.pt` | ultralytics | No | Attention vs convolution at matched capacity |
| `yolov13n` | `yolov13n.pt` | iMoonLab fork | No | Newest CNN; hypergraph feature correlation |

**Do not substitute models.** No RT-DETR, RF-DETR, or YOLO-NAS. RT-DETR variants are 8–10x larger in parameters and would confound architecture with capacity. YOLO-NAS is unmaintained since Deci's acquisition by NVIDIA and its default weight URLs are broken.

Optional 7th arm, only if the first six complete with quota to spare: `yolo26n-p2` via `yolo26n-p2.yaml`, testing whether a P2 high-resolution branch recovers recall on distant bumps.

## PHASE 0 — DATASET, BUILT ONCE, FROZEN FOREVER

The dataset is prepared **locally, once**, then uploaded as a **versioned Kaggle Dataset**. Every training kernel mounts that exact dataset version. This is not a convenience: it is what makes the split provably identical across all 18 runs, which a lock-file queue would otherwise have to enforce.

Acquire real data. Do not synthesize or fabricate annotations.

1. Roboflow Universe `cse400b` "Pothole & Speed Breaker-400B" — ~3,003 images, CC BY 4.0
2. Roboflow Universe "Humps/Bumps & Potholes" (`detection-system`) — ~3,243 images, CC BY 4.0
3. Mendeley Data DOI `10.17632/bvpt9xdjz8.1` — 969 images, Indian roads, marked speed breakers

**Verify the Mendeley set's annotation format before merging.** It may contain cropped classification images rather than bounding boxes. If so, exclude it and log the exclusion.

Processing steps:
- Merge to a **single class `speed_bump`**. Map or drop pothole annotations, and record which you chose.
- Deduplicate near-identical frames with perceptual hashing.
- **Split at source-video or session level, never at frame level.** Consecutive frames from one approach to one bump are effectively the same image; if they straddle the split, mAP measures memorization and the study is void. Where session metadata is absent, cluster by perceptual hash and split by cluster.
- Target 70/15/15.
- Write `train.txt`, `val.txt`, `test.txt` and a `split_manifest.json` containing the SHA-256 of each file list.

Upload via `kaggle datasets create` / `version`. Record the dataset slug and version number in `configs/dataset.yaml`. **Halt and report if the split audit detects cross-split leakage.**

## PHASE 1 — KERNEL GENERATION

Generate **one self-contained notebook per (model, seed)**, from a Jinja template. Each kernel must:

1. Mount the frozen dataset by exact slug and version.
2. Install its framework in-kernel. Two mutually exclusive paths, because YOLOv13 ships a vendored fork of Ultralytics that will clobber the stock package:
   - `yolo26n`, `yolo11n`, `yolov8n`, `yolov10n`, `yolo12n` → pinned stock `ultralytics`
   - `yolov13n` → `pip install -e` from `github.com/iMoonLab/yolov13` (needs internet enabled; FlashAttention is optional, proceed without it if the build fails and record that)
3. Train with the fixed protocol below.
4. Write to `/kaggle/working`: best weights, `results.csv`, full stdout log, `model.info()` output (params and GFLOPs measured, **not** copied from published tables), and a `run_manifest.json` recording resolved package versions, GPU model, driver, CUDA version, dataset version, seed, and the split manifest hash.
5. Export best weights to ONNX for later edge measurement.

Emit `kernel-metadata.json` per kernel with `enable_gpu: true`, `enable_internet: true`, and a deterministic slug like `sb-bench-{model}-seed{n}`.

## TRAINING PROTOCOL — HELD CONSTANT

- Input resolution **640 x 640**
- Epochs **150**, early stopping patience 50
- Batch size: the largest that fits **all six** models on the chosen accelerator, then fixed for every run
- COCO-pretrained initialization from each model's official weights
- `close_mosaic` for the final 10 epochs
- Vertical flip **disabled** (`flipud=0.0`) — a vertically mirrored road scene destroys the perspective cue that distinguishes a bump from a shadow band
- Horizontal flip, HSV jitter, scale, translate: framework defaults
- Seeds **0, 1, 2**

**Deliberate exception, documented as a limitation:** each model uses its own repository's native optimizer and schedule (MuSGD for YOLO26, SGD for the others). Forcing a foreign recipe onto a model handicaps it unfairly. What must be identical is the data, split, resolution, epoch budget, accelerator type, and evaluation script. Write this distinction into the README so it lifts directly into the paper's limitations section.

## PHASE 2 — QUOTA-AWARE EXECUTION

Budget: 18 runs at roughly 45–70 minutes each on a T4 is **20–30 hours**, which is the entire weekly quota with no margin. YOLOv12n and YOLOv13n will run longer than the others.

Execute in waves, not all at once:

- **Wave 1 (~6–8 h):** all six models at seed 0. This alone yields a publishable single-seed comparison.
- **Wave 2 (next quota week):** seeds 1 and 2.

Before each wave, query and log remaining quota. If a wave would exceed it, **stop and report rather than starting runs that will be killed mid-training.**

**Parallelism rule.** You may submit at most the number of concurrent GPU kernels Kaggle actually permits (verify; historically ~2). Concurrent *training* kernels are acceptable because each has its own session and accuracy metrics are unaffected. **Latency kernels must run strictly alone.** A latency measurement taken while another job holds the accelerator is worthless, and latency is a primary reported metric here, not a secondary one.

Write `scripts/submit.py` (push + record kernel slug), `scripts/poll.py` (`kaggle kernels status` until complete, with backoff), and `scripts/collect.py` (`kaggle kernels output` into `results/runs/<model>/<seed>/`). Never poll faster than once per minute.

Handle failure explicitly: a killed or errored kernel is logged as a failure and resubmitted at most once. **Never substitute a published COCO figure or an estimate for a run that did not complete.**

## PHASE 3 — EVALUATION, ONE SHARED PATH

Run evaluation **locally**, after collecting weights and predictions, through a single code path for every model.

Do not call each framework's own `val()` and collect the printed numbers. Different repos apply different confidence floors, max-detection caps, and IoU matching details, and mixing them silently invalidates the comparison. This is the most common way a benchmark paper gets dismantled in review.

Metrics per model per seed:
- Precision, Recall
- mAP@0.5, mAP@0.5:0.95
- **AP_small — report this prominently.** Speed bumps at useful stopping distance are often under 16 px tall in a 640 px frame; aggregate mAP masks exactly the behaviour this study exists to measure.
- Parameters and GFLOPs from the measured `model.info()` output

Report mean and standard deviation across seeds. Nano models on a ~3,000-image dataset vary enough between runs that a single seed can flip the ranking; a single-seed number is provisional and must be labelled as such.

## PHASE 4 — LATENCY, AND ITS LIMITS

Report **three separate numbers**: preprocessing, inference, post-processing. Never a single fused figure.

The model set deliberately mixes NMS-free models (`yolo26n`, `yolov10n`) with NMS-based ones. NMS cost scales with the number of raw candidate boxes, which rises under exactly the false-positive pressure this domain produces: zebra crossings, shadow bands, wet asphalt reflections. A fused number would attribute that cost to inference and make the NMS-free claim unfalsifiable.

**GPU latency (Kaggle, valid):** a dedicated kernel, nothing else running, 50 warmup frames discarded, 500 measured frames, reporting mean, standard deviation, and p95. Tail latency breaks a real-time loop; mean latency does not capture it.

**CPU / edge latency (Kaggle, NOT valid):** Kaggle's CPU is shared and virtualized. Numbers measured there do not represent a Raspberry Pi 5 or a Jetson and must not be reported as edge latency. Emit the ONNX exports and a standalone `scripts/edge_latency.py` that runs on real hardware with no training dependencies. Until that hardware measurement exists, mark every edge-latency cell as **not measured** rather than filling it from Kaggle.

For `yolo26n`, record both the NMS-free operating point and the one-to-many head, since the NMS-free path is documented to trade roughly 0.6–0.8 mAP for its latency saving. Reporting both makes the trade-off visible instead of assumed.

## OUTPUTS

1. `results/tables/accuracy.csv` and `.tex` — model, P, R, mAP@0.5, mAP@0.5:0.95, AP_small, mean ± std
2. `results/tables/efficiency.csv` and `.tex` — model, params, GFLOPs, preprocess / inference / postprocess ms, total, p95, FPS, with edge columns marked not-measured until real hardware data exists
3. `results/figures/` — accuracy-vs-latency scatter (the key figure for an edge-deployment argument), PR curves, stacked latency decomposition bar
4. `results/environment.json` — every run's resolved versions, GPU model, dataset version
5. `results/quota_log.csv` — quota consumed per wave
6. `README.md` — exact commands to reproduce every number

LaTeX tables in IEEE conference format using `booktabs`, single-column width where they fit.

## SELF-AUDIT

After the sweep, verify and emit `results/audit.json`:
- All runs used the identical dataset slug **and version**
- All `split_manifest.json` hashes match
- Identical resolution and epoch budget across runs
- No run reported as complete without a corresponding collected output directory
- Latency runs recorded as having run alone

**If the audit fails, the results are not usable. Surface that prominently and do not proceed to table generation.**

## EXECUTION ORDER

1. Local environment, dependency pinning, Kaggle CLI auth check
2. Dataset acquisition, verification, dedup, split
3. **Split leakage audit — halt on failure**
4. Dataset upload, version recorded and frozen
5. Kernel template and per-run notebook generation
6. Wave 1 submission (6 models, seed 0), poll, collect
7. Shared evaluation, provisional single-seed tables
8. Wave 2 (seeds 1–2) when quota resets
9. GPU latency kernels, run alone
10. Aggregation, tables, figures, self-audit

Report progress after each stage. Halt and tell me if stage 3 or the self-audit fails, or if remaining quota is insufficient for the next wave.
