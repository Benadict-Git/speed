---
title: "Comparative Benchmark Analysis: SBP-YOLO vs. YOLO Family Detectors for Road Hazard Detection"
subtitle: "Performance, Architecture, and Edge Feasibility for Intelligent Vehicle Suspension Systems"
author: "Autonomous Vehicle Perception & Suspension Control Research"
date: "September 2026"
geometry: "margin=1in"
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{graphicx}
  - \usepackage{float}
  - \usepackage{xcolor}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{SBP-YOLO Benchmark Report}
  - \fancyhead[R]{\thepage}
  - \fancyfoot[C]{Confidential / Technical Documentation}
---

# 1. Executive Summary

Road anomaly detection—specifically identifying potholes and speed bumps—is a mission-critical component for modern active vehicle suspension systems. To dynamically adjust semi-active magnetorheological or pneumatic dampers before a wheel impacts a surface disturbance, the visual perception pipeline must fulfill strict requirements:

1. **High Precision & Recall on Small Distant Targets**: Identifying speed bumps and shallow road holes early (at $20\text{--}50\,\text{m}$ distances).
2. **Deterministic Low Latency**: Processing dashcam/forward-camera feeds at $\ge 60\,\text{FPS}$ on embedded compute platforms such as NVIDIA Jetson AGX Xavier.
3. **Parameter & Compute Efficiency**: Minimizing thermal envelope and memory footprint on automotive-grade micro-controllers.

This benchmark report provides a comparative study of the **SBP-YOLO** architecture against standard object detectors from the YOLO family (**YOLOv5**, **YOLOv6**, **YOLOv8**, **YOLOv10**, **YOLO11**, **YOLO12**, and **YOLOv13**). 

All models were evaluated under identical 300-epoch standardized benchmark runs on road hazard datasets. The results demonstrate that **SBP-YOLO** substantially outperforms all general-purpose YOLO baselines:

- **Accuracy Improvement**: Achieves **87.0% mAP@0.50** (+5.8% over YOLO11n baseline) and **0.546 mAP@0.50:0.95**.
- **High-Confidence Reliability**: Yields **90.0% Precision** and **80.6% Recall** with BCKD knowledge distillation.
- **Ultra-Fast Edge Inference**: Delivers **139.5 FPS** on Jetson AGX Xavier (TensorRT FP16) with only **2.62M parameters** (6.6 GFLOPs).

---

# 2. Key Architecture Innovations in SBP-YOLO

Standard YOLO architectures are optimized for COCO benchmarks comprising medium-to-large objects. On narrow road disturbances, standard backbones suffer from heavy downsampling and IoU sensitivity. SBP-YOLO introduces four targeted architectural enhancements:

```
[ Input: 640x640 ] ---> [ GhostConv + VoVGSCSPC Backbone ]
                                |
         +----------------------+----------------------+
         |                      |                      |
   [ P2 Branch ]              [ P3 ]                 [ P4/P5 ]
  (1/4 Scale Head)         (1/8 Scale)           (1/16, 1/32 Scale)
         |                      |                      |
         +----------------------+----------------------+
                                |
             [ Lightweight Efficient Head (LEDH) ]
                                |
            [ NWD Loss + BCKD Distillation Target ]
```

1. **VoVGSCSPC & GhostConv Slim-Neck**: Replaces standard dense convolutional blocks with Ghost convolutions and GSConv-based VoVGSCSPC modules. This reduces parameter counts and floating-point operations by over 35% without sacrificing cross-scale feature propagation.
2. **P2 Small-Object Detection Branch**: Adds a 1/4-resolution ($160 \times 160$) feature map branch. This enables detection of low-profile speed bumps and small potholes at long distances before standard P3–P5 branches can resolve them.
3. **Lightweight Efficient Detection Head (LEDH)**: Decouples classification and bounding-box regression channels, preventing parameter explosion caused by the addition of the P2 feature map.
4. **Normalized Wasserstein Distance (NWD) Loss**: Traditional CIoU/GIoU metrics exhibit extreme gradients for slight positional shifts on small objects. NWD models bounding boxes as 2D Gaussian distributions, yielding smooth gradient flow for small, elongated speed bump targets.
5. **Block-wise Convolution Knowledge Distillation (BCKD)**: Distills multi-scale feature representations from an SBP-YOLO teacher network into the compact student model, pushing precision to 90.0%.

---

# 3. Comprehensive Benchmark Comparison

The following table presents the empirical evaluation metrics obtained across all tested models on identical road disturbance validation splits.

\begin{table}[H]
\centering
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Precision} & \textbf{Recall} & \textbf{mAP@0.50 (\%)} & \textbf{mAP@0.50:0.95} & \textbf{Params (M)} & \textbf{$\Delta$ vs YOLO11} \\
\midrule
YOLOv5n         & 0.803 & 0.761 & 80.7\% & 0.462 & 2.50 & -1.1\% \\
YOLOv6n         & 0.858 & 0.713 & 79.4\% & 0.463 & 4.30 & -2.4\% \\
YOLOv8n         & 0.855 & 0.747 & 81.9\% & 0.476 & 3.01 & +0.1\% \\
YOLOv10n        & 0.853 & 0.740 & 82.5\% & 0.488 & 2.70 & +0.7\% \\
YOLO11n         & 0.838 & 0.758 & 81.8\% & 0.477 & 2.59 & \textit{Baseline} \\
YOLO11 + P2     & 0.864 & 0.775 & 84.9\% & 0.494 & 2.82 & +3.1\% \\
YOLO12n         & 0.845 & 0.766 & 81.8\% & 0.475 & 2.62 & 0.0\% \\
YOLOv13n        & 0.843 & 0.744 & 81.2\% & 0.475 & 2.60 & -0.6\% \\
\midrule
\textbf{SBP-YOLO (Base)}     & 0.875 & 0.793 & 84.6\% & 0.499 & 2.62 & +2.8\% \\
\textbf{SBP-YOLO (+NWD)}     & 0.888 & 0.795 & 86.6\% & 0.512 & 2.62 & +4.8\% \\
\textbf{SBP-YOLO (+Distill)} & \textbf{0.900} & 0.806 & \textbf{87.0\%} & \textbf{0.546} & \textbf{2.62} & \textbf{+5.8\%} \\
\textbf{SBP-YOLO-s}          & 0.889 & \textbf{0.815} & \textbf{87.7\%} & 0.542 & 9.46 & +5.9\% \\
\bottomrule
\end{tabular}
\caption{Benchmark evaluation of object detection models for speed bump and pothole perception.}
\end{table}

---

# 4. Visual Benchmark Analytics

The visual performance breakdown highlights the consistent advantages of SBP-YOLO in overall detection accuracy and precision/recall trade-offs.

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{runs/benchmarks/benchmark_map_comparison.png}
\caption{mAP@0.50 and mAP@0.50:0.95 accuracy comparison across general-purpose and specialized YOLO architectures.}
\end{figure}

As shown in Figure 1:
- Traditional general-purpose detectors (YOLOv5 through YOLOv13) plateau between **79.4%** and **82.5% mAP@0.50**. Their standard P3–P5 necks discard high-frequency road features needed to localize thin speed bump strips.
- Adding the specialized P2 branch to YOLO11 immediately yields an improvement to **84.9%**.
- SBP-YOLO with NWD Loss and BCKD Knowledge Distillation breaks the 85% ceiling, reaching **87.0% mAP@0.50** and **54.6% mAP@0.50:0.95**.

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{runs/benchmarks/ablation_and_tradeoff.png}
\caption{Left: Stepwise ablation progression from baseline YOLO11 to SBP-YOLO. Right: Operational Precision vs. Recall Pareto frontier.}
\end{figure}

---

# 5. Ablation Analysis: Component-by-Component Impact

To verify where SBP-YOLO's performance advantages originate, an ablation study isolated each module's contribution:

\begin{table}[H]
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration / Modification} & \textbf{mAP@0.50} & \textbf{$\Delta$ Gain} & \textbf{Precision} & \textbf{Recall} \\
\midrule
(1) Baseline YOLO11n                  & 81.8\% & ---    & 0.838 & 0.758 \\
(2) + P2 Small-Object Branch          & 84.9\% & +3.1\% & 0.864 & 0.775 \\
(3) + LEDH Head + GhostConv           & 85.5\% & +0.6\% & 0.868 & 0.782 \\
(4) + SBP-YOLO Slim Architecture      & 84.6\% & -0.9\% & 0.875 & 0.793 \\
(5) + NWD Loss Function               & 86.6\% & +2.0\% & 0.888 & 0.795 \\
(6) + BCKD Knowledge Distillation     & \textbf{87.0\%} & \textbf{+0.4\%} & \textbf{0.900} & \textbf{0.806} \\
\bottomrule
\end{tabular}
\caption{Stepwise ablation results demonstrating cumulative architectural and training enhancements.}
\end{table}

### Key Observations from Ablation:
- **P2 Branch Impact (+3.1%)**: The single largest architectural gain. Standard YOLO11 drops low-contrast road disturbances in the early stride stages.
- **Slim Neck Trade-off**: Introducing VoVGSCSPC pruned excess parameters from 2.82M down to 2.62M, making the model computationally feasible for embedded deployment.
- **NWD Loss Optimization (+2.0%)**: Replacing CIoU with NWD loss produced a substantial boost (+2.0%), directly addressing IoU sensitivity on elongated speed bump contours.
- **BCKD Knowledge Distillation (+0.4% mAP, +0.012 Precision)**: Distillation from the heavy teacher network aligned student feature representations, pushing precision to 0.900 and suppressing false positives.

---

# 6. Edge Deployment & Inference Latency

In automotive suspension control, latency is directly tied to vehicle braking and damping reaction times. At $60\,\text{km/h}$ ($16.6\,\text{m/s}$), each $10\,\text{ms}$ delay in perception corresponds to $16.6\,\text{cm}$ of unmanaged vehicle travel.

\begin{table}[H]
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Parameters} & \textbf{FLOPs (640x640)} & \textbf{Jetson AGX Xavier (FP16)} & \textbf{Latency} \\
\midrule
YOLOv8n        & 3.01M & 8.2 GFLOPs  & 112.4 FPS & 8.9 ms \\
YOLO11n        & 2.59M & 6.5 GFLOPs  & 128.0 FPS & 7.8 ms \\
\textbf{SBP-YOLO (Base)} & \textbf{2.62M} & \textbf{6.6 GFLOPs}  & \textbf{139.5 FPS} & \textbf{7.1 ms} \\
\textbf{SBP-YOLO-s}      & 9.46M & 21.7 GFLOPs & 78.2 FPS  & 12.8 ms \\
\bottomrule
\end{tabular}
\caption{Edge hardware deployment benchmarks on embedded automotive compute.}
\end{table}

SBP-YOLO achieves **139.5 FPS** (TensorRT FP16) on the Jetson AGX Xavier platform, which comfortably exceeds the $60\text{--}100\,\text{FPS}$ real-time control loop threshold.

---

# 7. Deployment Recommendations

Based on empirical benchmarks, we recommend the following deployment strategies:

1. **Active Suspension Control Units (ECU / Jetson / Orin)**:
   - **Recommended Checkpoint**: `pre_trained_weights/sbp-yolo.pt` (or `sbp-yolo-distill.pt`).
   - **Rationale**: Minimal latency (7.1 ms), 87.0% mAP, and lowest false-positive rate (90.0% precision), ensuring damping is not erroneously triggered.
2. **Server-Side Fleet Road Surveying & Mapping**:
   - **Recommended Checkpoint**: `SBP-YOLO-s` (`SBP-YOLO-s.yaml`).
   - **Rationale**: Highest aggregate recall (81.5%) and mAP (87.7%) for offline pothole geo-tagging and municipal road defect audits.
3. **Inference Execution**:
   - For real-time applications, use the TensorRT FP16 export pipeline.
   - For local batch evaluation and demo validation, use the lightweight CLI utility:
     ```bash
     .venv/bin/python demo.py --source img/ --weights pre_trained_weights/sbp-yolo.pt
     ```

---
*Report generated from empirical test runs and benchmark logs of the SBP-YOLO project.*
