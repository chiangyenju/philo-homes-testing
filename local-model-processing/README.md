# 🏠 Room Object Removal Pipeline

A comprehensive solution for removing objects from room images using state-of-the-art AI models. This pipeline provides multiple inpainting models to choose from, each optimized for different use cases.

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   python setup.py
   ```

2. **Generate Masks** 
   ```bash
   python room_removal_ultimate.py
   ```
   - Upload your room image
   - Detect and select objects to remove
   - Generate precise masks
   - Export as zip file

3. **Choose Your Inpainting Model**
   ```bash
   # LaMa - Best overall quality
   python lama_inpainting.py
   
   # MAT - Advanced transformer-based
   python mat_inpainting.py
   
   # SDXL - Creative/artistic results
   python sdxl_inpainting.py
   
   # SD 1.5 - Fast and lightweight
   python sd15_inpainting.py
   ```

## 📋 Overview

This pipeline consists of **5 main scripts**:

### 1. 🎯 `room_removal_ultimate.py` - Mask Generation
- **YOLO v8** object detection
- **SAM (Segment Anything)** precise segmentation  
- Interactive selection of objects to remove
- Exports image + mask as zip file

### 2. 🎨 `lama_inpainting.py` - LaMa Model
- **Best overall quality** for object removal
- Multi-scale refinement (CUDA required)
- Excellent for architectural/interior scenes
- Uses Large Mask Inpainting model

### 3. 🎭 `mat_inpainting.py` - MAT Model  
- **Mask-Aware Transformer** architecture
- Advanced attention mechanisms
- Good for complex scenes
- Multiple pretrained checkpoints

### 4. 🌟 `sdxl_inpainting.py` - SDXL Model
- **Stable Diffusion XL** inpainting
- Creative and artistic results
- Prompt-guided generation
- Best for stylistic transformations

### 5. ⚡ `sd15_inpainting.py` - SD 1.5 Model
- **Fast and lightweight**
- Stable Diffusion 1.5 based
- Good balance of speed/quality
- Automatic image resizing for optimal performance

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.7+ (recommended for GPU acceleration)
- 16GB+ RAM (32GB recommended for SDXL)
- 20GB+ free disk space

### Automatic Setup
```bash
python setup.py
```

## 📁 Project Structure

```
local-model-processing/
├── room_removal_ultimate.py    # Mask generation
├── lama_inpainting.py          # LaMa inpainting  
├── mat_inpainting.py           # MAT inpainting
├── sdxl_inpainting.py          # SDXL inpainting
├── sd15_inpainting.py          # SD 1.5 inpainting
├── setup.py                    # Setup script
├── README.md                   # This file
├── models/                     # Downloaded models
├── exports/                    # Generated zip files
└── results/                    # Final results
```

## 🎯 Workflow

### Step 1: Mask Generation
1. Run `python room_removal_ultimate.py`
2. Upload your room image
3. Adjust detection parameters and select objects
4. Export as zip file

### Step 2: Choose Inpainting Model

#### 🎨 LaMa - Recommended for Most Cases
- **Best for**: Architectural interiors, clean removal
- **Strengths**: Natural texture completion, multi-scale refinement

#### 🎭 MAT - Advanced Transformer
- **Best for**: Complex scenes, detailed textures  
- **Strengths**: Attention-based inpainting

#### 🌟 SDXL - Creative Results
- **Best for**: Artistic transformations, creative fills
- **Strengths**: Prompt-guided generation, high resolution

#### ⚡ SD 1.5 - Fast & Lightweight  
- **Best for**: Quick results, limited hardware
- **Strengths**: Fast inference, automatic resizing

## ⚙️ Model Comparison

| Model | Quality | Speed | VRAM | Best For |
|-------|---------|-------|------|----------|
| **LaMa** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 4GB | Architectural |
| **MAT** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 6GB | Complex scenes |
| **SDXL** | ⭐⭐⭐⭐⭐ | ⭐⭐ | 8GB | Artistic |
| **SD 1.5** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 4GB | Quick results |

## 🔧 Usage Tips

### For Best Quality:
- Use LaMa with official refinement
- SAM vit_h model with mask dilation 15-20px
- YOLO confidence 0.001 for maximum detection

### For Speed:
- Use SD 1.5 model
- SAM vit_b model
- Reduce inference steps to 20-30

### For Limited VRAM:
- Use SD 1.5 instead of SDXL
- Process images at 512px max resolution

## 🚨 Troubleshooting

**"CUDA out of memory"**: Reduce image resolution or use CPU mode
**"Model not found"**: Run `python setup.py` again
**"LaMa refinement not working"**: AMD GPU users need CUDA for refinement

## 🤝 Acknowledgments

This pipeline combines multiple open-source projects:
- [YOLO v8](https://github.com/ultralytics/ultralytics) - Object detection
- [Segment Anything](https://github.com/facebookresearch/segment-anything) - Segmentation  
- [LaMa](https://github.com/advimman/lama) - Large Mask Inpainting
- [MAT](https://github.com/fenglinglwb/MAT) - Mask-Aware Transformer
- [Stable Diffusion](https://huggingface.co/docs/diffusers) - Diffusion models
