"""
Download modern inpainting models for Google Drive storage
Run this once to download all models, then upload to Google Drive
"""
import os
import requests
from tqdm import tqdm
import hashlib

def download_file(url, filepath, expected_size=None):
    """Download file with progress bar"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    if expected_size and os.path.getsize(filepath) < expected_size * 0.9:
        print(f"⚠️ Warning: {filepath} seems incomplete")
    
    return filepath

def download_sdxl_models():
    """
    Download SDXL inpainting models
    Best quality, modern approach
    """
    print("\n📥 Downloading SDXL Inpainting Models...")
    print("Note: These are large files (~5-6GB each)")
    
    models = {
        # SDXL Base model for inpainting
        "sdxl_inpainting/vae/config.json": {
            "url": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/vae/config.json",
            "size": 0.001  # 1KB
        },
        "sdxl_inpainting/vae/diffusion_pytorch_model.safetensors": {
            "url": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/vae/diffusion_pytorch_model.safetensors",
            "size": 335  # 335MB
        },
        "sdxl_inpainting/unet/config.json": {
            "url": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/unet/config.json",
            "size": 0.002  # 2KB
        },
        "sdxl_inpainting/unet/diffusion_pytorch_model.safetensors": {
            "url": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/unet/diffusion_pytorch_model.safetensors",
            "size": 5135  # 5.1GB
        },
        "sdxl_inpainting/model_index.json": {
            "url": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/model_index.json",
            "size": 0.001  # 1KB
        }
    }
    
    for filepath, info in models.items():
        target_path = f"models/{filepath}"
        if os.path.exists(target_path):
            print(f"✅ Already exists: {filepath}")
        else:
            print(f"⬇️ Downloading: {filepath} (~{info['size']}MB)")
            download_file(info['url'], target_path, info['size'] * 1024 * 1024)

def download_controlnet_models():
    """
    Download ControlNet inpainting models
    Excellent for edge preservation
    """
    print("\n📥 Downloading ControlNet Inpainting Models...")
    
    models = {
        "controlnet_inpaint/config.json": {
            "url": "https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/resolve/main/config.json",
            "size": 0.001  # 1KB
        },
        "controlnet_inpaint/diffusion_pytorch_model.safetensors": {
            "url": "https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/resolve/main/diffusion_pytorch_model.safetensors",
            "size": 1445  # 1.4GB
        }
    }
    
    for filepath, info in models.items():
        target_path = f"models/{filepath}"
        if os.path.exists(target_path):
            print(f"✅ Already exists: {filepath}")
        else:
            print(f"⬇️ Downloading: {filepath} (~{info['size']}MB)")
            download_file(info['url'], target_path, info['size'] * 1024 * 1024)

def download_mat_model():
    """
    Download MAT (Mask-Aware Transformer) model
    Modern LaMa alternative
    """
    print("\n📥 Downloading MAT Model...")
    
    # MAT model from Google Drive (you'll need to use gdown)
    print("MAT model requires manual download:")
    print("1. Go to: https://drive.google.com/file/d/1HMnQhkqr6qTXXHmMNu5hpBLs8JvK6JVw")
    print("2. Download MAT_Places512.pkl")
    print("3. Place in models/mat/MAT_Places512.pkl")
    
    os.makedirs("models/mat", exist_ok=True)

def create_upload_script():
    """
    Create script to upload models to Google Drive
    """
    script = '''#!/bin/bash
# Upload models to Google Drive

echo "Uploading inpainting models to Google Drive..."
echo "Make sure you have Google Drive mounted first!"

# Create directory structure in Google Drive
mkdir -p "/content/drive/MyDrive/inpainting_models"

# Copy SDXL models
echo "Copying SDXL models..."
cp -r models/sdxl_inpainting "/content/drive/MyDrive/inpainting_models/"

# Copy ControlNet models
echo "Copying ControlNet models..."
cp -r models/controlnet_inpaint "/content/drive/MyDrive/inpainting_models/"

# Copy MAT model if available
if [ -f "models/mat/MAT_Places512.pkl" ]; then
    echo "Copying MAT model..."
    cp -r models/mat "/content/drive/MyDrive/inpainting_models/"
fi

echo "✅ Upload complete!"
echo "Models are now in: /content/drive/MyDrive/inpainting_models/"
'''
    
    with open("upload_to_drive.sh", "w") as f:
        f.write(script)
    
    print("\n📝 Created upload_to_drive.sh")
    print("Run this in Colab after mounting Drive to upload models")

def print_summary():
    """Print summary of model sizes and usage"""
    print("\n" + "="*60)
    print("📊 MODEL SUMMARY")
    print("="*60)
    
    print("""
Model Sizes:
- SDXL Inpainting: ~6GB total (best quality)
- ControlNet: ~1.5GB (good edge preservation)
- MAT: ~500MB (fast, good quality)

Total Space Needed: ~8GB

Google Colab Usage:
1. Upload these models to Google Drive once
2. Mount Drive in Colab
3. Load models from Drive (no re-downloading)
4. Models stay cached between sessions

Quality Comparison:
- SDXL: ★★★★★ (Photorealistic, best overall)
- ControlNet: ★★★★☆ (Great structure preservation)
- MAT: ★★★★☆ (Fast, reliable)
- LaMa: ★★★☆☆ (Older, dependency issues)
""")

if __name__ == "__main__":
    print("🚀 Modern Inpainting Models Downloader")
    print("="*60)
    
    # Check available space
    import shutil
    free_space = shutil.disk_usage(".").free / (1024**3)
    print(f"💾 Free disk space: {free_space:.1f}GB")
    
    if free_space < 10:
        print("⚠️ Warning: Less than 10GB free space!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit()
    
    # Download models
    print("\nWhich models to download?")
    print("1. SDXL Inpainting (6GB) - RECOMMENDED")
    print("2. ControlNet Inpainting (1.5GB)")
    print("3. Both (7.5GB)")
    print("4. Skip download, just create upload script")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == "1":
        download_sdxl_models()
    elif choice == "2":
        download_controlnet_models()
    elif choice == "3":
        download_sdxl_models()
        download_controlnet_models()
    
    download_mat_model()  # Instructions only
    create_upload_script()
    print_summary()
    
    print("\n✅ Done! Next steps:")
    print("1. Upload models to Google Drive")
    print("2. Use the Colab notebooks with your Drive models")
    print("3. No more downloading on each Colab run!")