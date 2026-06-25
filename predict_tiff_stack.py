# ============================================================
# YOLOv8 Organoid Segmentation — Predict on TIFF Stack
# Streams frames one at a time to avoid OOM on large stacks
# Reads a multi-page TIFF, runs inference on each frame,
# and saves masks + optionally annotated frames as output.

# Usage: 
# Fast mode (masks only — for viability analysis)
# python predict_tiff_stack.py \
#   --tiff path-to-your-tiff-file \
#   --model path-to-your_best.pt \
#   --outdir your-output-folder-path \
#   --masks_only

# Full mode (masks + annotated — for visual QC)
# python predict_tiff_stack.py \
#   --tiff path-to-your-tiff-file \
#   --model path-to-your_yolov8l-seg.pt \
#   --outdir your-output-folder-path
# ============================================================

import argparse
import numpy as np
import tifffile
from pathlib import Path
from ultralytics import YOLO

# ── Arguments ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run YOLOv8 segmentation on a TIFF stack")
parser.add_argument("--tiff",       required=True,  help="Path to input multi-page TIFF file")
parser.add_argument("--model",      required=True,  help="Path to model weights (best.pt)")
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

# ── Output directory ─────────────────────────────────────────
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

tiff_path = Path(args.tiff)

# ── Stream TIFF frames one at a time ─────────────────────────
# Uses TiffFile page iterator to avoid loading entire stack into RAM
print(f"\nOpening TIFF stack: {tiff_path}")

with tifffile.TiffFile(str(tiff_path)) as tif:
    n_frames = len(tif.pages)
    print(f"  Total frames: {n_frames}")

    # Get image shape from first page
    first_page = tif.pages[0].asarray()
    H, W = first_page.shape[:2]
    print(f"  Frame size: {H} x {W}")

    # Pre-allocate output mask stack as memory-mapped file
    mask_tiff_path = outdir / f"{tiff_path.stem}_masks.tif"
    annotated_tiff_path = outdir / f"{tiff_path.stem}_annotated.tif"

    # Open output writers
    mask_writer = tifffile.TiffWriter(str(mask_tiff_path), bigtiff=True)
    if not args.masks_only:
        annotated_writer = tifffile.TiffWriter(str(annotated_tiff_path), bigtiff=True)

    print(f"\nRunning inference on {n_frames} frames...")

    for i, page in enumerate(tif.pages):
        # Read single frame
        frame = page.asarray()

        # Normalize to uint8 if needed
        if frame.dtype != np.uint8:
            frame_min = frame.min()
            frame_max = frame.max()
            if frame_max > frame_min:
                frame_norm = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
            else:
                frame_norm = np.zeros_like(frame, dtype=np.uint8)
        else:
            frame_norm = frame

        # Convert grayscale to RGB
        frame_rgb = np.stack([frame_norm] * 3, axis=-1)

        # Run inference
        results = model.predict(
            source=frame_rgb,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )

        result = results[0]
        n_detected = len(result.boxes) if result.boxes is not None else 0

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Frame {i+1:05d}/{n_frames:05d} — {n_detected} organoids detected")

        # Write binary mask
        if result.masks is not None:
            combined_mask = result.masks.data.cpu().numpy().max(axis=0)
            combined_mask = (combined_mask * 255).astype(np.uint8)
        else:
            combined_mask = np.zeros((H, W), dtype=np.uint8)
        mask_writer.write(combined_mask)

        # Write annotated frame if needed
        if not args.masks_only:
            annotated = result.plot()
            annotated_writer.write(annotated)

    # Close writers
    mask_writer.close()
    if not args.masks_only:
        annotated_writer.close()

print(f"\nSaved mask stack -> {mask_tiff_path}")
if not args.masks_only:
    print(f"Saved annotated stack -> {annotated_tiff_path}")
print("\nDone.")