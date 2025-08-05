#!/usr/bin/env python3
"""Quick setup - download only missing YOLO models"""

import os
import sys
from ultralytics import YOLO

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

print("Quick Model Setup")
print("=" * 50)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Download YOLO models using ultralytics (it will save to models/)
yolo_models = ["yolov8x.pt", "yolov8l.pt", "yolov8m.pt"]

for model_name in yolo_models:
    local_path = f"models/{model_name}"
    if os.path.exists(local_path):
        size = os.path.getsize(local_path) / (1024 * 1024)
        print(f"✅ {model_name} already exists ({size:.1f} MB)")
    else:
        print(f"📥 Downloading {model_name}...")
        try:
            # This will download to the models/ directory
            model = YOLO(local_path)
            print(f"✅ Downloaded {model_name}")
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")

print("\n" + "=" * 50)
print("Model Status:")
print("-" * 50)

# Check all models
model_files = {
    "yolov8x.pt": "Best object detection",
    "yolov8l.pt": "Large object detection", 
    "sam_vit_h_4b8939.pth": "Best segmentation",
    "sam_vit_b_01ec64.pth": "Fast segmentation",
    "best.ckpt": "Big-LaMa model",
    "config.yaml": "LaMa config"
}

missing = []
for model_file, desc in model_files.items():
    path = f"models/{model_file}"
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ {model_file:<25} ({size:>7.1f} MB) - {desc}")
    else:
        print(f"❌ {model_file:<25} (missing) - {desc}")
        missing.append(model_file)

if "best.ckpt" in missing or "config.yaml" in missing:
    print("\n⚠️ Big-LaMa model missing!")
    print("Download from: https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip")
    print("Extract to models/ directory")

print("\n✨ Setup complete! Run: python room_removal_ultimate.py")