# Modern Inpainting Setup Guide

## Quick Start After Download

### Step 1: Download Models (On Fast Internet)

```bash
cd local-model-processing
python download_inpainting_models.py
```

Choose one:
- **Option 1**: SDXL (5.5GB) - Best quality ⭐⭐⭐⭐⭐
- **Option 2**: SD 1.5 (2GB) - Good quality ⭐⭐⭐⭐
- **Option 3**: Skip (use existing LaMa) ⭐⭐⭐

### Step 2: Upload to Google Drive

After download completes:

1. **Create folder in Google Drive:**
   ```
   Google Drive/My Drive/inpainting_models/
   ```

2. **Upload the downloaded model folder:**
   - If SDXL: Upload `models/sdxl_inpainting/` folder
   - If SD 1.5: Upload `models/sd15_inpaint/` folder

3. **Keep folder structure intact:**
   ```
   Google Drive/
   └── My Drive/
       └── inpainting_models/
           └── sdxl_inpainting/  (or sd15_inpaint/)
               ├── vae/
               │   ├── config.json
               │   └── diffusion_pytorch_model.safetensors
               ├── unet/
               │   ├── config.json
               │   └── diffusion_pytorch_model.safetensors
               └── model_index.json
   ```

### Step 3: Your New Workflow

#### Local Processing (AMD GPU Compatible)
1. Run `room_removal_ultimate.py`
2. Upload image → Detect objects → Create masks
3. Click "Export for Colab Processing"
4. You'll get a zip file with image + mask

#### Google Colab Processing (High Quality)
1. Open `sdxl_drive_colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run cells in order:
   - Mount Google Drive (your models are there)
   - Load SDXL from Drive (no downloading!)
   - Upload your exported zip
   - Get photorealistic inpainting results

## File Descriptions

### Main Files You'll Use

| File | Purpose | When to Use |
|------|---------|------------|
| `room_removal_ultimate.py` | Detect & mask objects | Always (locally) |
| `sdxl_drive_colab.ipynb` | SDXL inpainting | Best quality (Colab) |
| `download_inpainting_models.py` | Download models once | Initial setup only |

### Alternative Options

| File | Purpose | Quality |
|------|---------|---------|
| `lama_simple_colab.ipynb` | Old LaMa method | ⭐⭐⭐ (dependency issues) |
| `controlnet_inpaint_colab.ipynb` | Edge-preserving | ⭐⭐⭐⭐ |
| `mat_inpainting.py` | LaMa alternative | ⭐⭐⭐⭐ |

## Complete Example Workflow

### 1. Local (5 minutes)
```bash
python room_removal_ultimate.py
```
- Upload room photo
- Detect objects (YOLO)
- Select what to remove
- Create masks (SAM)
- Export for Colab → `colab_export/[timestamp].zip`

### 2. Google Colab (2 minutes)
- Open `sdxl_drive_colab.ipynb`
- Run all cells
- Upload your zip
- Download result

### 3. Result
- Original: Room with furniture
- Output: Empty room, photorealistic

## Model Comparison

| Model | Size | Quality | Speed | Recommendation |
|-------|------|---------|-------|----------------|
| **SDXL** | 5.5GB | ⭐⭐⭐⭐⭐ | Medium | **Best quality** |
| SD 1.5 | 2GB | ⭐⭐⭐⭐ | Fast | Space-conscious |
| LaMa | 200MB | ⭐⭐⭐ | Fast | Already have |
| ControlNet | 1.5GB | ⭐⭐⭐⭐ | Fast | Good edges |

## Troubleshooting

### "Models not found in Drive"
- Make sure path is: `/My Drive/inpainting_models/`
- Check folder structure matches above

### "Out of memory in Colab"
- Use Runtime → Restart runtime
- Make sure GPU is enabled
- Try SD 1.5 instead of SDXL

### "Poor quality results"
- Increase mask dilation in room_removal_ultimate.py
- Try multiple inference steps in Colab
- Use SDXL instead of SD 1.5

## Why This Is Better Than LaMa

1. **Works**: No dependency conflicts
2. **Quality**: Photorealistic vs painted look
3. **Modern**: Works with latest PyTorch
4. **Cached**: Models stay in Drive, no re-downloading
5. **Flexible**: Text-guided inpainting

## Storage Requirements

- **Google Drive Free**: 15GB total
- **SDXL Model**: 5.5GB
- **Plenty of space** for results and other files

## Next Steps After Setup

1. ✅ Models in Google Drive
2. ✅ Test with one image
3. ✅ Adjust prompts for your room style
4. ✅ Batch process multiple rooms

## Tips for Best Results

- **Mask Dilation**: Set to 15-20px to include shadows
- **Prompt**: "empty room, professional real estate photo"
- **Negative**: "furniture, objects, artifacts"
- **Multi-pass**: Run 2 passes for highest quality

---

**Ready to go!** Your workflow is now:
1. Local: Detect + Mask (AMD GPU works)
2. Colab: Inpaint (NVIDIA GPU, cached models)
3. Result: Photorealistic empty rooms