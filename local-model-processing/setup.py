#!/usr/bin/env python3
"""
Setup Script for Room Object Removal Pipeline
Downloads models and installs dependencies
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e.stderr}")
        return False

def download_file(url, filepath, description=""):
    """Download a file with progress"""
    print(f"📥 Downloading {description}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ Downloaded {description}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {description}: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    dirs = [
        "models",
        "models/sdxl_inpainting",
        "models/sd15_inpaint", 
        "mat_workspace",
        "exports",
        "results"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✅ {dir_path}")

def install_python_packages():
    """Install required Python packages"""
    print("🐍 Installing Python packages...")
    
    packages = [
        "torch torchvision",
        "gradio",
        "opencv-python",
        "pillow",
        "numpy",
        "ultralytics",
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        "diffusers",
        "transformers", 
        "accelerate",
        "omegaconf",
        "pyyaml"
    ]
    
    success = True
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            success = False
    
    return success

def download_yolo_models():
    """Download YOLO models"""
    print("🎯 Downloading YOLO models...")
    
    models = {
        "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
        "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"
    }
    
    success = True
    for model_name, url in models.items():
        filepath = f"models/{model_name}"
        if not os.path.exists(filepath):
            if not download_file(url, filepath, f"YOLO {model_name}"):
                success = False
        else:
            print(f"✅ {model_name} already exists")
    
    return success

def download_sam_models():
    """Download SAM models"""
    print("✂️ Downloading SAM models...")
    
    models = {
        "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
        "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    success = True
    for model_name, url in models.items():
        filepath = f"models/{model_name}"
        if not os.path.exists(filepath):
            if not download_file(url, filepath, f"SAM {model_name}"):
                success = False
        else:
            print(f"✅ {model_name} already exists")
    
    return success

def setup_lama():
    """Set up LaMa model"""
    print("🎨 Setting up LaMa model...")
    
    # Download LaMa checkpoint and config
    lama_files = {
        "best.ckpt": "https://github.com/advimman/lama/releases/download/main/big-lama.zip",
        "config.yaml": None  # Will be included in the zip
    }
    
    # Download LaMa code
    saicinpainting_path = "models/saicinpainting"
    if not os.path.exists(saicinpainting_path):
        print("📥 Cloning LaMa repository...")
        if run_command(f"git clone https://github.com/advimman/lama.git temp_lama", "Cloning LaMa"):
            # Copy saicinpainting module
            if os.path.exists("temp_lama/saicinpainting"):
                shutil.copytree("temp_lama/saicinpainting", saicinpainting_path)
                print("✅ LaMa saicinpainting module copied")
            
            # Clean up
            if os.path.exists("temp_lama"):
                shutil.rmtree("temp_lama")
    
    # Download LaMa checkpoint
    checkpoint_path = "models/best.ckpt"
    if not os.path.exists(checkpoint_path):
        zip_path = "models/big-lama.zip"
        if download_file("https://github.com/advimman/lama/releases/download/main/big-lama.zip", 
                        zip_path, "LaMa checkpoint"):
            # Extract zip
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("models/")
                os.remove(zip_path)
                print("✅ LaMa checkpoint extracted")
            except Exception as e:
                print(f"❌ Failed to extract LaMa checkpoint: {e}")
                return False
    else:
        print("✅ LaMa checkpoint already exists")
    
    return True

def setup_mat():
    """Set up MAT model"""
    print("🎭 Setting up MAT model...")
    
    mat_path = "mat_workspace/MAT"
    if not os.path.exists(mat_path):
        print("📥 Cloning MAT repository...")
        if not run_command(f"git clone https://github.com/fenglinglwb/MAT.git {mat_path}", "Cloning MAT"):
            return False
    
    # Create pretrained directory
    pretrained_path = f"{mat_path}/pretrained"
    os.makedirs(pretrained_path, exist_ok=True)
    
    print("⚠️ MAT pretrained models need to be downloaded manually:")
    print("  1. Visit: https://github.com/fenglinglwb/MAT#pretrained-models")
    print("  2. Download desired models (CelebA-HQ, Places2, etc.)")
    print(f"  3. Place them in: {pretrained_path}/")
    
    return True

def download_diffusion_models():
    """Set up Stable Diffusion models"""
    print("🌟 Setting up Stable Diffusion models...")
    
    try:
        from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipeline
        
        # Download SDXL inpainting model
        sdxl_path = "models/sdxl_inpainting"
        if not os.path.exists(f"{sdxl_path}/model_index.json"):
            print("📥 Downloading SDXL inpainting model...")
            pipeline = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                variant="fp16"
            )
            pipeline.save_pretrained(sdxl_path)
            print("✅ SDXL inpainting model downloaded")
        else:
            print("✅ SDXL inpainting model already exists")
        
        # Download SD 1.5 inpainting model
        sd15_path = "models/sd15_inpaint"
        if not os.path.exists(f"{sd15_path}/model_index.json"):
            print("📥 Downloading SD 1.5 inpainting model...")
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting"
            )
            pipeline.save_pretrained(sd15_path)
            print("✅ SD 1.5 inpainting model downloaded")
        else:
            print("✅ SD 1.5 inpainting model already exists")
        
        return True
        
    except ImportError:
        print("⚠️ Diffusers not available - Stable Diffusion models will be downloaded on first use")
        return True

def check_gpu():
    """Check GPU availability"""
    print("🖥️ Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU detected: {gpu_name}")
            return "cuda"
        else:
            try:
                import torch_directml
                print("✅ DirectML detected - AMD GPU support available")
                return "directml"
            except ImportError:
                print("⚠️ No GPU acceleration available - using CPU")
                return "cpu"
    except ImportError:
        print("❌ PyTorch not installed")
        return None

def main():
    """Main setup function"""
    print("🏠 Room Object Removal - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        return False
    
    # Create directories
    setup_directories()
    
    # Install packages
    if not install_python_packages():
        print("❌ Failed to install some Python packages")
        return False
    
    # Check GPU
    gpu_type = check_gpu()
    
    # Download models
    print("\n📦 Downloading models...")
    
    success = True
    success &= download_yolo_models()
    success &= download_sam_models()
    success &= setup_lama()
    success &= setup_mat()
    
    # Download diffusion models (optional, can be slow)
    print("\n🌟 Diffusion models setup (optional - can download on first use):")
    user_input = input("Download SDXL and SD 1.5 models now? (y/N): ").lower().strip()
    if user_input in ['y', 'yes']:
        download_diffusion_models()
    else:
        print("⚠️ Diffusion models will be downloaded automatically on first use")
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Setup completed successfully!")
        print("\n🚀 You can now run:")
        print("  python room_removal_ultimate.py  - Generate masks")
        print("  python lama_inpainting.py       - LaMa inpainting")
        print("  python mat_inpainting.py        - MAT inpainting")
        print("  python sdxl_inpainting.py       - SDXL inpainting")
        print("  python sd15_inpainting.py       - SD 1.5 inpainting")
    else:
        print("⚠️ Setup completed with some errors")
        print("Check the output above for details")
    
    return success

if __name__ == "__main__":
    main()