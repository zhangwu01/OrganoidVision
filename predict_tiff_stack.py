# ============================================================
# YOLOv8 Organoid Segmentation — Predict on TIFF Stack
# Reads a multi-page TIFF, runs inference on each frame,
# and saves masks + optionally annotated frames as output.

# Usage: 
# Fast mode (masks only — for viability analysis)
# python predict_tiff_stack.py \
#   --tiff stack.tif \
#   --model yolov8l-seg.pt \
#   --masks_only

# Full mode (masks + annotated — for visual QC)
# python predict_tiff_stack.py \
#   --tiff stack.tif \
#   --model yolov8l-seg.pt
# ============================================================

import argparse
import numpy as np
import tifffile
from pathlib import Path
from ultralytics import YOLO

# ── Arguments ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run YOLOv8 segmentation on a TIFF stack")
parser.add_argument("--tiff",       required=True,  help="Path to input multi-page TIFF file")
parser.add_argument("--model",      required=True,  help="Path to model weights (yolov8l-seg.pt)")
parser.add_argument("--outdir",     default="predictions", help="Output directory")
parser.add_argument("--conf",       type=float, default=0.25, help="Confidence threshold")
parser.add_argument("--device",     default="0",    help="Device: 0 (GPU), mps, or cpu")
parser.add_argument("--masks_only", action="store_true",
                    help="Skip annotated TIFF output — only save binary masks (faster)")
args = parser.parse_args()

# ── Load model ───────────────────────────────────────────────
print(f"Loading model: {args.model}")
model = YOLO(args.model)
print(f"Model loaded successfully")

if args.masks_only:
    print("masks_only mode — annotated TIFF will be skipped")

# ── Load TIFF stack ──────────────────────────────────────────
tiff_path = Path(args.tiff)
print(f"\nLoading TIFF stack: {tiff_path}")

stack = tifffile.imread(str(tiff_path))

if stack.ndim == 2:
    stack = stack[np.newaxis, ...]
print(f"  Stack shape: {stack.shape}  ({stack.shape[0]} frames)")

# ── Output directory ─────────────────────────────────────────
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

# ── Run inference frame by frame ─────────────────────────────
annotated_stack = []
mask_stack      = []

print(f"\nRunning inference on {stack.shape[0]} frames...")

for i, frame in enumerate(stack):
    # Normalize to uint8 if needed (TIFF frames can be 16-bit)
    if frame.dtype != np.uint8:
        frame_norm = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)
    else:
        frame_norm = frame

    # Convert grayscale to RGB (YOLOv8 expects 3-channel input)
    frame_rgb = np.stack([frame_norm] * 3, axis=-1)

    results = model.predict(
        source=frame_rgb,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )

    result = results[0]
    n_detected = len(result.boxes) if result.boxes is not None else 0
    print(f"  Frame {i+1:04d}/{stack.shape[0]:04d} — {n_detected} organoids detected")

    # Annotated frame — only if not masks_only
    if not args.masks_only:
        annotated = result.plot()
        annotated_stack.append(annotated)

    # Binary mask (always saved)
    if result.masks is not None:
        combined_mask = result.masks.data.cpu().numpy().max(axis=0)
        combined_mask = (combined_mask * 255).astype(np.uint8)
    else:
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_stack.append(combined_mask)

# ── Save outputs ─────────────────────────────────────────────
print("\nSaving outputs...")

# Mask TIFF stack (always saved)
mask_tiff = outdir / f"{tiff_path.stem}_masks.tif"
tifffile.imwrite(str(mask_tiff), np.stack(mask_stack).astype(np.uint8))
print(f"  Mask stack      -> {mask_tiff}")

# Annotated TIFF stack (only if not masks_only)
if not args.masks_only:
    annotated_tiff = outdir / f"{tiff_path.stem}_annotated.tif"
    tifffile.imwrite(str(annotated_tiff), np.stack(annotated_stack).astype(np.uint8))
    print(f"  Annotated stack -> {annotated_tiff}")
else:
    print("  Annotated stack — skipped (masks_only mode)")

print("\nDone.")
