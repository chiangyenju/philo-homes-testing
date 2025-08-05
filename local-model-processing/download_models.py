#!/usr/bin/env python3
"""
One-time model download script for Ultimate Room Object Removal
Downloads all required models to avoid runtime downloads
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import hashlib

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def download_file(url, filepath, expected_size=None):
    """Download a file with progress indicator"""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"  Progress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='\r')
    
    print(f"Downloading {filepath.name}...")
    urllib.request.urlretrieve(url, filepath, reporthook=download_progress)
    print()  # New line after progress
    
    if expected_size and filepath.stat().st_size != expected_size:
        print(f"⚠️ Warning: File size mismatch for {filepath.name}")

def main():
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("🚀 Ultimate Room Object Removal - Model Downloader")
    print("=" * 50)
    
    # Model URLs and info
    models = {
        "yolov8x.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
            "size": 136719344,  # Expected size in bytes
            "description": "YOLOv8x - Best object detection"
        },
        "yolov8l.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
            "size": 87792836,
            "description": "YOLOv8l - Large object detection"
        },
        "yolov8m.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
            "size": 52015476,
            "description": "YOLOv8m - Medium object detection"
        },
        "sam_vit_h_4b8939.pth": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "size": 2564550879,
            "description": "SAM ViT-H - Best segmentation quality"
        },
        "sam_vit_l_0b3195.pth": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
            "size": 1249524607,
            "description": "SAM ViT-L - Large segmentation model"
        },
        "sam_vit_b_01ec64.pth": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "size": 375042383,
            "description": "SAM ViT-B - Fast segmentation"
        }
    }
    
    # Check and download models
    for filename, info in models.items():
        filepath = models_dir / filename
        
        if filepath.exists():
            actual_size = filepath.stat().st_size
            if actual_size == info["size"]:
                print(f"✅ {filename} already exists ({actual_size:,} bytes)")
                continue
            else:
                print(f"⚠️ {filename} exists but size mismatch. Re-downloading...")
                filepath.unlink()
        
        print(f"\n📥 {info['description']}")
        try:
            download_file(info["url"], filepath, info["size"])
            print(f"✅ Downloaded {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
    
    # Check for Big-LaMa model
    print("\n" + "=" * 50)
    print("📦 Checking Big-LaMa model...")
    
    big_lama_files = ["best.ckpt", "config.yaml"]
    lama_complete = all((models_dir / f).exists() for f in big_lama_files)
    
    if lama_complete:
        print("✅ Big-LaMa model already installed")
        ckpt_size = (models_dir / "best.ckpt").stat().st_size
        print(f"   best.ckpt: {ckpt_size:,} bytes")
    else:
        print("\n⚠️ Big-LaMa model not found!")
        print("To install Big-LaMa (required for best results):")
        print("1. Download from: https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip")
        print("2. Extract big-lama.zip to the models/ directory")
        print("3. Ensure you have:")
        print("   - models/best.ckpt")
        print("   - models/config.yaml")
    
    # Check saicinpainting module
    saicinpainting_path = models_dir / "saicinpainting"
    if saicinpainting_path.exists():
        print("\n✅ LaMa inference module (saicinpainting) found")
    else:
        print("\n⚠️ LaMa inference module not found!")
        print("To enable LaMa:")
        print("1. git clone https://github.com/advimman/lama.git")
        print("2. Copy the 'saicinpainting' folder to models/")
    
    print("\n" + "=" * 50)
    print("✨ Model setup complete!")
    print("\nYou can now run: python room_removal_ultimate.py")
    print("All models will load from local files without downloading.")

if __name__ == "__main__":
    main()