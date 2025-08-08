# Modern Inpainting Setup Guide

## Quick Start

### Step 1: Download Models (On Fast Internet)

```bash
cd local-model-processing
python download_all_models.py
```

Choose one:
- **Option 1**: SDXL (5.5GB) - Best quality ⭐⭐⭐⭐⭐
- **Option 2**: SD 1.5 (2GB) - Good quality ⭐⭐⭐⭐
- **Option 3**: Both models
- **Option 4**: Resume incomplete downloads

### Step 2: Upload to Google Drive

After download completes:

1. **Create folder in Google Drive:**
   ```
   Google Drive/My Drive/inpainting_models/
   ```

2. **Upload the downloaded model folder(s):**
   - SDXL: Upload `models/sdxl_inpainting/` folder
   - SD 1.5: Upload `models/sd15_inpaint/` folder

3. **Final structure in Drive:**
   ```
   My Drive/
   └── inpainting_models/
       ├── sdxl_inpainting/     # 5.5GB total
       │   ├── unet/
       │   │   ├── config.json
       │   │   └── diffusion_pytorch_model.safetensors
       │   ├── vae/
       │   │   ├── config.json
       │   │   └── diffusion_pytorch_model.safetensors
       │   └── model_index.json
       └── sd15_inpaint/        # 2GB total
           ├── unet/
           ├── vae/
           ├── text_encoder/
           ├── tokenizer/
           ├── scheduler/
           └── model_index.json
   ```

### Step 3: Your Workflow

#### Local Processing (Works on AMD GPU)
```bash
python room_removal_ultimate.py
```
1. Upload room photo
2. Detect objects (YOLO)
3. Select what to remove
4. Create masks (SAM)
5. Export for Colab → saves `colab_export/[timestamp].zip`

#### Google Colab Processing
1. Open **`inpainting_colab.ipynb`** in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Mount Google Drive (has your models)
4. Change `MODEL_CHOICE = "sdxl"` or `"sd15"`
5. Upload your exported zip
6. Get photorealistic results!

## Files You Actually Need

| File | Purpose | When to Use |
|------|---------|------------|
| **`room_removal_ultimate.py`** | Detect & mask objects | Always run locally first |
| **`inpainting_colab.ipynb`** | Universal inpainting notebook | Use in Google Colab |
| **`download_all_models.py`** | Download models once | Initial setup only |
| `export_for_colab.py` | Export utility | Auto-used by room_removal |

## Model Comparison

| Model | Size | Quality | Speed | Use When |
|-------|------|---------|-------|----------|
| **SDXL** | 5.5GB | ⭐⭐⭐⭐⭐ | Slower | Want best quality |
| **SD 1.5** | 2GB | ⭐⭐⭐⭐ | Faster | Good enough + faster |
| LaMa | 200MB | ⭐⭐⭐ | Fast | Basic inpainting |

## Complete Example

### 1. First Time Setup (One Time Only)
```bash
# Download models
python download_all_models.py
# Choose Option 1 (SDXL) or 2 (SD 1.5)

# Upload to Google Drive manually
# Path: My Drive/inpainting_models/
```

### 2. Regular Use
```bash
# Local: Detect and mask
python room_removal_ultimate.py
# → Export → Creates colab_export/timestamp.zip

# Colab: Inpaint
# Open inpainting_colab.ipynb
# Change MODEL_CHOICE = "sdxl"
# Upload zip → Get result
```

## Troubleshooting

### "Models not found in Drive"
- Check path: `/content/drive/MyDrive/inpainting_models/`
- Make sure folder names match exactly

### "Download interrupted"
```bash
python download_all_models.py
# Choose Option 4 to resume
```

### "Out of memory in Colab"
- Use SD 1.5 instead of SDXL
- Or add `pipe.enable_model_cpu_offload()`

### "Poor quality results"
- Use SDXL instead of SD 1.5
- Increase mask dilation to 20px
- Better prompt: "high quality empty room, professional real estate photography"

## Direct Download Links (If Script Fails)

### SDXL Files (Total 5.5GB):
1. **UNet** (5.1GB): [Download](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/unet/diffusion_pytorch_model.safetensors)
   - Save to: `models/sdxl_inpainting/unet/`

2. **VAE** (335MB): [Download](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/vae/diffusion_pytorch_model.safetensors)
   - Save to: `models/sdxl_inpainting/vae/`

3. Config files (small): Download from same base URL

### SD 1.5 Files (Total 2GB):
- Main model files from: `https://huggingface.co/runwayml/stable-diffusion-inpainting/tree/main`

## Tips for Best Results

- **Mask Dilation**: 15-20px to include shadows
- **SDXL Prompts**: "high quality interior, empty room, professional real estate photography"
- **Negative Prompts**: "furniture, objects, people, artifacts, blurry"
- **Inference Steps**: 50 for best quality
- **Multi-pass**: Run twice for refinement

## Why This Setup?

✅ **No dependency issues** - Modern models work with latest PyTorch  
✅ **Cached in Drive** - Download once, use forever  
✅ **Photorealistic** - SDXL/SD1.5 >> LaMa quality  
✅ **One notebook** - `inpainting_colab.ipynb` handles everything  
✅ **AMD compatible** - Detection/masking works locally  

---

**Quick Reference:**
1. Local: `room_removal_ultimate.py` → Export zip
2. Colab: `inpainting_colab.ipynb` → Upload zip → Get result
3. Models in Drive = No re-downloading!