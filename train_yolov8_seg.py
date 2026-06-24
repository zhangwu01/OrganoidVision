# ============================================================
# YOLOv8 Instance Segmentation -- Roboflow Dataset
# Export format: YOLOv8 PyTorch (Segmentation)
# Model: Transfer Learning (COCO pretrained)
# ============================================================

from roboflow import Roboflow
from ultralytics import YOLO
import pandas as pd
import torch
import os

# Auto-detect device
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    DEVICE = 0 if n_gpus == 1 else list(range(n_gpus))
    print(f"Using CUDA GPU(s): {DEVICE}  ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple Silicon MPS")
else:
    DEVICE = "cpu"
    print("No GPU found -- falling back to CPU")

# Image size and batch: scale by device 
# M1/M2 Mac (9GB MPS limit): use smaller imgsz and batch
# HPC GPU (A100/V100 40-80GB): use full resolution
if DEVICE == "mps":
    IMGSZ = 640
    BATCH  = 4
    print(f"M1 Mac detected: imgsz={IMGSZ}, batch={BATCH}")
else:
    IMGSZ = int(os.environ.get("TRAIN_IMGSZ", 1280))
    BATCH  = int(os.environ.get("TRAIN_BATCH", 16))
    print(f"HPC/GPU detected: imgsz={IMGSZ}, batch={BATCH}")

# Download dataset from Roboflow 
rf = Roboflow(api_key="PL9WhqPzdYDJyrqYlMi4")
project = rf.workspace("zhangwu-uchicago-edu").project("organoid-clo9p")
version = project.version(1)

dataset = version.download("yolov8")
DATA_YAML = f"{dataset.location}/data.yaml"

# Shared training arguments 
TRAIN_ARGS = dict(
    data=DATA_YAML,
    epochs=150,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=30,
    device=DEVICE,
    workers=0 if DEVICE == "mps" else 8,  # MPS works better with 0 workers
    save=True,
    plots=True,
    # Brightfield augmentation (grayscale, rotation-invariant)
    degrees=180.0,
    fliplr=0.5,
    flipud=0.5,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.3,
    amp=True,
)

# Model Training: Transfer Learning (COCO pretrained) 
print("\n" + "="*60)
print("Training Model: Transfer Learning (yolov8l-seg.pt)")
print("="*60)

model = YOLO("yolov8l-seg.pt")

model.train(
    **TRAIN_ARGS,
    project="organoid_seg",
    name="modelA_transfer_learning",
)

metrics = model.val()
map_box  = metrics.box.map
map_mask = metrics.seg.map
print(f"Box  mAP50-95: {map_box:.4f}")
print(f"Mask mAP50-95: {map_mask:.4f}")
