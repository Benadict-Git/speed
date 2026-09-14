#!/usr/bin/env python3
"""
SBP-YOLO Demo Script
Run inference on images, video files, directories, or a webcam stream.

Usage examples:
    python demo.py --source img/
    python demo.py --source /path/to/road_video.mp4
    python demo.py --source 0                      # webcam
    python demo.py --weights pre_trained_weights/sbp-yolo.pt --conf 0.35
"""

import sys
import argparse
from pathlib import Path

# Ensure the ultralytics-distill fork is prioritized for custom SBP-YOLO modules
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "ultralytics-distill"))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run SBP-YOLO Demo")
    parser.add_argument(
        "--weights",
        type=str,
        default="pre_trained_weights/sbp-yolo.pt",
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="img/",
        help="Source path: image file, directory, video file, or camera index (e.g. 0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on: 'cpu' or '0' for CUDA GPU",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory to save results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="demo",
        help="Experiment name for saved results",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Display live results window during inference (OpenCV)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file '{weights_path}' not found.")
        sys.exit(1)

    print(f"Loading SBP-YOLO model from: {weights_path}")
    model = YOLO(str(weights_path))
    print(f"Detected classes: {model.names}")

    # Convert source to int if numeric (webcam index)
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"Running inference on source: {source} (device={args.device}, conf={args.conf})")
    results = model.predict(
        source=source,
        conf=args.conf,
        device=args.device,
        save=True,
        show=args.view,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print(f"\nInference complete! Results saved to: {Path(args.project) / args.name}")
    for r in results:
        boxes = r.boxes
        print(f"\nSource: {r.path}")
        if len(boxes) == 0:
            print("  No potholes or speed bumps detected.")
        for b in boxes:
            cls_name = model.names[int(b.cls.item())]
            conf = float(b.conf.item())
            box = [round(x, 1) for x in b.xyxy.tolist()[0]]
            print(f"  - {cls_name} (conf={conf:.2f}) at {box}")


if __name__ == "__main__":
    main()
