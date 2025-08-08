"""
Smaller, lighter inpainting models as alternatives to SDXL
"""
import os
import requests
from tqdm import tqdm

def download_file(url, filepath):
    """Download with progress bar"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
            for chunk in response.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))

def option1_sd15_inpainting():
    """
    Stable Diffusion 1.5 Inpainting
    Size: ~2GB total (much smaller than SDXL)
    Quality: Still very good, just not as detailed as SDXL
    """
    print("\n📥 Downloading SD 1.5 Inpainting (2GB total)...")
    
    base_url = "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/"
    
    files = {
        "vae/config.json": 0.001,
        "vae/diffusion_pytorch_model.bin": 335,  # 335MB
        "unet/config.json": 0.002,
        "unet/diffusion_pytorch_model.bin": 1720,  # 1.7GB
        "model_index.json": 0.001,
    }
    
    for file, size_mb in files.items():
        target = f"models/sd15_inpaint/{file}"
        if os.path.exists(target):
            print(f"✅ Exists: {file}")
        else:
            print(f"⬇️ Downloading: {file} (~{size_mb}MB)")
            download_file(base_url + file, target)
    
    print("✅ SD 1.5 Inpainting ready! (2GB total)")

def option2_lama_only():
    """
    Just use your existing LaMa model
    Size: 200MB (you already have this)
    Quality: Good for simple inpainting, no refinement
    """
    print("\n💡 Using existing LaMa model (200MB)")
    print("You already have this in models/best.ckpt")
    print("Quality: ★★★☆☆ (decent but not photorealistic)")
    
def option3_opencv_inpaint():
    """
    OpenCV traditional inpainting
    Size: 0MB (built into OpenCV)
    Quality: Poor but works instantly
    """
    print("\n💡 OpenCV Inpainting (No download needed)")
    print("Already included in OpenCV")
    print("Quality: ★☆☆☆☆ (basic, not recommended)")

if __name__ == "__main__":
    print("🎯 Smaller Model Options")
    print("="*50)
    print("\nChoose based on your constraints:")
    print("\n1. SD 1.5 Inpainting (2GB) - Good quality, 60% smaller")
    print("2. Keep using LaMa (200MB) - You have this")
    print("3. Cancel SDXL, use SD 1.5 instead")
    print("4. Continue with SDXL (best quality)")
    
    choice = input("\nChoice (1-4): ")
    
    if choice == "1" or choice == "3":
        # Cancel SDXL download if running
        print("\n⚠️ Cancel the SDXL download (Ctrl+C)")
        print("Then downloading SD 1.5 instead...")
        option1_sd15_inpainting()
    elif choice == "2":
        option2_lama_only()
    elif choice == "4":
        print("\n✅ Continue with SDXL download")
        print("It's worth it for the quality!")