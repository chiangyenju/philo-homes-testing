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

def download_mat():
    """Download MAT (Mask-Aware Transformer) models"""
    print("\n=== MAT Inpainting Models ===")
    print("State-of-the-art transformer-based inpainting")
    print("Better quality than LaMa, works with modern PyTorch")
    
    os.makedirs("models/mat/pretrained", exist_ok=True)
    
    # MAT models with OneDrive links
    models = {
        "Places512": {
            "filename": "models/mat/pretrained/Places_512_FullData.pkl",
            "url": "https://mycuhk-my.sharepoint.com/:u:/g/personal/1155137927_link_cuhk_edu_hk/EYzXPHvdFGlOl_o0p0moCsEBmj7E8wEPxRSMqmX0N2uF0A?download=1",
            "size_mb": 1300,
            "description": "Best Places model (8M training images)"
        },
        "CelebA-HQ": {
            "filename": "models/mat/pretrained/CelebA-HQ.pkl",
            "url": "https://mycuhk-my.sharepoint.com/:u:/g/personal/1155137927_link_cuhk_edu_hk/ETTRAiTzzJpNv00y8h_FqN8BwU5tv8A2w-uqW4FmrhYpCQ?download=1",
            "size_mb": 1300,
            "description": "Face inpainting model"
        },
        "FFHQ-512": {
            "filename": "models/mat/pretrained/FFHQ.pkl", 
            "url": "https://mycuhk-my.sharepoint.com/:u:/g/personal/1155137927_link_cuhk_edu_hk/ESwt5gvPs4JOvC76WAEDfb4BSJZNy-qsfJSUZz2kTxYyWw?download=1",
            "size_mb": 1300,
            "description": "High-quality face inpainting"
        }
    }
    
    print("\nAvailable MAT models:")
    for name, info in models.items():
        status = "✓" if os.path.exists(info["filename"]) else "✗"
        print(f"  {status} {name}: {info['description']}")
    
    print("\n" + "="*60)
    print("MAT MODELS REQUIRE MANUAL DOWNLOAD")
    print("="*60)
    print("\nDue to OneDrive restrictions, MAT models need manual download.")
    print("\nOption 1: Download from OneDrive links")
    print("Places512 (Best): https://bit.ly/mat-places512")
    print("CelebA-HQ: https://bit.ly/mat-celebahq") 
    print("FFHQ-512: https://bit.ly/mat-ffhq")
    
    print("\nOption 2: All models folder")
    print("https://mycuhk-my.sharepoint.com/:f:/g/personal/1155137927_link_cuhk_edu_hk/EuY30ziF-G5BvwziuHNFzDkBVC6KBPRg69kCeHIu-BXORA")
    
    print("\nSave downloaded models to:")
    print("  models/mat/pretrained/")
    
    # Clone MAT repository if not present
    mat_repo = "models/mat/MAT"
    if not os.path.exists(mat_repo):
        print("\nCloning MAT repository...")
        try:
            import subprocess
            result = subprocess.run(
                ["git", "clone", "https://github.com/fenglinglwb/MAT.git", mat_repo],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("  ✓ MAT repository cloned")
            else:
                print(f"  ✗ Failed to clone: {result.stderr}")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    else:
        print("\n  ✓ MAT repository already exists")

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
    
    # Check MAT models
    mat_models = ["Places_512_FullData.pkl", "CelebA-HQ.pkl", "FFHQ.pkl"]
    mat_found = False
    for model in mat_models:
        path = f"models/mat/pretrained/{model}"
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  MAT {model}: {size_mb:.1f}MB")
            mat_found = True
    if not mat_found:
        print(f"  MAT: No models found")
    
    return sdxl_complete

def main():
    print("="*60)
    print("COMPREHENSIVE INPAINTING MODEL DOWNLOADER")
    print("="*60)
    
    # Check what exists
    sdxl_complete = check_existing()
    
    print("\n=== Options ===")
    print("1. Download SDXL (5.5GB) - Best quality diffusion")
    print("2. Download SD 1.5 (2GB) - Good quality, smaller")
    print("3. Download both SDXL and SD 1.5")
    print("4. Setup MAT models - State-of-the-art transformer")
    print("5. Resume/fix incomplete SDXL")
    print("6. Check status only")
    
    choice = input("\nChoice (1-6): ").strip()
    
    if choice == "1":
        download_sdxl()
    elif choice == "2":
        download_sd15()
    elif choice == "3":
        download_sdxl()
        download_sd15()
    elif choice == "4":
        download_mat()
    elif choice == "5":
        print("\nResuming SDXL downloads...")
        download_sdxl()
    elif choice == "6":
        print("\nStatus check complete.")
        return
    else:
        print("Invalid choice")
        return
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. For SDXL/SD1.5 - Upload to Google Drive:")
    print("   - models/sdxl_inpainting/ -> My Drive/inpainting_models/sdxl_inpainting/")
    print("   - models/sd15_inpaint/ -> My Drive/inpainting_models/sd15_inpaint/")
    print("   - Use Colab notebooks: sdxl_drive_colab.ipynb, sd15_colab.ipynb")
    
    print("\n2. For MAT models:")
    print("   - Download models manually from OneDrive links provided")
    print("   - Place in models/mat/pretrained/")
    print("   - Use mat_inpainting.py for setup and integration")
    
    print("\nModel Comparison:")
    print("• SDXL: Best photorealistic quality, large files (5.5GB)")
    print("• SD 1.5: Good quality, smaller files (2GB)")  
    print("• MAT: State-of-the-art transformer, best structure preservation")
    
    print("\nWorkflow:")
    print("1. Local: room_removal_ultimate.py (detect + mask)")
    print("2. Export for Colab or local processing")
    print("3. Run inpainting with your chosen model")
    print("4. MAT generally gives best results for large hole inpainting!")

if __name__ == "__main__":
    main()