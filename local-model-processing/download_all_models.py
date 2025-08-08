"""
Comprehensive Model Downloader - All inpainting models in one script
Choose what you want to download
"""
import os
import requests
from tqdm import tqdm
import time

def download_file(url, filepath, resume=True):
    """Download with resume support"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    headers = {}
    mode = 'wb'
    resume_pos = 0
    
    # Check for resume
    if resume and os.path.exists(filepath):
        resume_pos = os.path.getsize(filepath)
        headers['Range'] = f'bytes={resume_pos}-'
        mode = 'ab'
        print(f"  Resuming from {resume_pos / 1024 / 1024:.1f}MB")
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            total_size = int(response.headers.get('content-length', 0)) + resume_pos
            
            with open(filepath, mode) as f:
                with tqdm(total=total_size, initial=resume_pos, unit='B', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Verify size
            final_size = os.path.getsize(filepath)
            if final_size < total_size * 0.95:
                print(f"  Warning: File may be incomplete ({final_size / 1024 / 1024:.1f}MB)")
            return True
            
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {str(e)[:50]}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print(f"  Failed to download {os.path.basename(filepath)}")
                return False

def download_sdxl():
    """Download SDXL Inpainting (5.5GB) - Best quality"""
    print("\n=== SDXL Inpainting (5.5GB) ===")
    print("Best quality, photorealistic results")
    
    files = {
        "models/sdxl_inpainting/unet/diffusion_pytorch_model.safetensors": {
            "url": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/unet/diffusion_pytorch_model.safetensors",
            "size_mb": 5135
        },
        "models/sdxl_inpainting/vae/diffusion_pytorch_model.safetensors": {
            "url": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/vae/diffusion_pytorch_model.safetensors",
            "size_mb": 335
        },
        "models/sdxl_inpainting/unet/config.json": {
            "url": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/unet/config.json",
            "size_mb": 0.002
        },
        "models/sdxl_inpainting/vae/config.json": {
            "url": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/vae/config.json",
            "size_mb": 0.001
        },
        "models/sdxl_inpainting/model_index.json": {
            "url": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/model_index.json",
            "size_mb": 0.001
        }
    }
    
    for filepath, info in files.items():
        # Check if already complete
        if os.path.exists(filepath):
            current_size = os.path.getsize(filepath) / 1024 / 1024
            if current_size >= info["size_mb"] * 0.95:
                print(f"  Skip: {os.path.basename(filepath)} already complete")
                continue
        
        print(f"\nDownloading: {os.path.basename(filepath)} ({info['size_mb']}MB)")
        download_file(info["url"], filepath)

def download_sd15():
    """Download SD 1.5 Inpainting (2GB) - Good quality, smaller"""
    print("\n=== SD 1.5 Inpainting (2GB) ===")
    print("Good quality, much smaller than SDXL")
    
    base_url = "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/"
    
    files = {
        "models/sd15_inpaint/unet/diffusion_pytorch_model.bin": {"path": "unet/diffusion_pytorch_model.bin", "size_mb": 1720},
        "models/sd15_inpaint/vae/diffusion_pytorch_model.bin": {"path": "vae/diffusion_pytorch_model.bin", "size_mb": 335},
        "models/sd15_inpaint/text_encoder/pytorch_model.bin": {"path": "text_encoder/pytorch_model.bin", "size_mb": 492},
        "models/sd15_inpaint/unet/config.json": {"path": "unet/config.json", "size_mb": 0.002},
        "models/sd15_inpaint/vae/config.json": {"path": "vae/config.json", "size_mb": 0.001},
        "models/sd15_inpaint/text_encoder/config.json": {"path": "text_encoder/config.json", "size_mb": 0.001},
        "models/sd15_inpaint/tokenizer/tokenizer_config.json": {"path": "tokenizer/tokenizer_config.json", "size_mb": 0.001},
        "models/sd15_inpaint/tokenizer/vocab.json": {"path": "tokenizer/vocab.json", "size_mb": 1.1},
        "models/sd15_inpaint/tokenizer/merges.txt": {"path": "tokenizer/merges.txt", "size_mb": 0.5},
        "models/sd15_inpaint/scheduler/scheduler_config.json": {"path": "scheduler/scheduler_config.json", "size_mb": 0.001},
        "models/sd15_inpaint/model_index.json": {"path": "model_index.json", "size_mb": 0.001}
    }
    
    for filepath, info in files.items():
        if os.path.exists(filepath):
            current_size = os.path.getsize(filepath) / 1024 / 1024
            if current_size >= info["size_mb"] * 0.95:
                print(f"  Skip: {os.path.basename(filepath)} already complete")
                continue
        
        print(f"\nDownloading: {os.path.basename(filepath)} ({info['size_mb']}MB)")
        download_file(base_url + info["path"], filepath)

def check_existing():
    """Check what's already downloaded"""
    print("\n=== Checking Existing Files ===")
    
    # Check SDXL
    sdxl_files = {
        "models/sdxl_inpainting/unet/diffusion_pytorch_model.safetensors": 5135,
        "models/sdxl_inpainting/vae/diffusion_pytorch_model.safetensors": 335
    }
    
    sdxl_complete = True
    for filepath, expected_mb in sdxl_files.items():
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            status = "Complete" if size_mb >= expected_mb * 0.95 else f"Incomplete ({size_mb:.1f}/{expected_mb}MB)"
            print(f"  SDXL {os.path.basename(filepath)}: {status}")
            if size_mb < expected_mb * 0.95:
                sdxl_complete = False
        else:
            print(f"  SDXL {os.path.basename(filepath)}: Not found")
            sdxl_complete = False
    
    # Check SD 1.5
    sd15_main = "models/sd15_inpaint/unet/diffusion_pytorch_model.bin"
    if os.path.exists(sd15_main):
        size_mb = os.path.getsize(sd15_main) / 1024 / 1024
        print(f"  SD1.5 main model: {size_mb:.1f}MB")
    else:
        print(f"  SD1.5: Not found")
    
    return sdxl_complete

def main():
    print("="*60)
    print("COMPREHENSIVE INPAINTING MODEL DOWNLOADER")
    print("="*60)
    
    # Check what exists
    sdxl_complete = check_existing()
    
    print("\n=== Options ===")
    print("1. Download SDXL (5.5GB) - Best quality")
    print("2. Download SD 1.5 (2GB) - Good quality, smaller")
    print("3. Download both")
    print("4. Resume/fix incomplete SDXL")
    print("5. Check status only")
    
    choice = input("\nChoice (1-5): ").strip()
    
    if choice == "1":
        download_sdxl()
    elif choice == "2":
        download_sd15()
    elif choice == "3":
        download_sdxl()
        download_sd15()
    elif choice == "4":
        print("\nResuming SDXL downloads...")
        download_sdxl()
    elif choice == "5":
        print("\nStatus check complete.")
        return
    else:
        print("Invalid choice")
        return
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Upload to Google Drive:")
    print("   - models/sdxl_inpainting/ -> My Drive/inpainting_models/sdxl_inpainting/")
    print("   - models/sd15_inpaint/ -> My Drive/inpainting_models/sd15_inpaint/")
    print("2. Use the Colab notebooks:")
    print("   - sdxl_drive_colab.ipynb (for SDXL)")
    print("   - sd15_colab.ipynb (for SD 1.5)")
    print("\nWorkflow:")
    print("1. Local: room_removal_ultimate.py (detect + mask)")
    print("2. Export for Colab")
    print("3. Colab: Run inpainting with your Drive models")

if __name__ == "__main__":
    main()