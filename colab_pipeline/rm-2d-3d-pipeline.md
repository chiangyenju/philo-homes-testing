# Furniture 2D to 3D Pipeline

This pipeline processes furniture images to create 3D models using Google Colab with the latest Hunyuan3D-2.1.

## Pipeline Steps

1. **Background Removal** - Uses RemBG/SAM2 to remove backgrounds from furniture images
2. **3D Model Generation** - Uses Hunyuan3D-2.1 to convert 2D images to 3D GLB models with PBR materials
3. **Cloud Storage** - Uploads models and textures to Google Drive or Firebase Storage
4. **Catalog Export** - Creates JSON catalog for web integration

## Setup Instructions

### 1. Prepare Your Environment

1. Upload the notebook to Google Colab
2. Enable GPU runtime: Runtime → Change runtime type → GPU
3. Prepare your furniture images in Google Drive

### 2. Configure Storage

#### Option A: Google Drive (Simple)
- Images are saved to `/content/drive/MyDrive/philo-homes-3d/`
- No additional configuration needed

#### Option B: Firebase Storage (Production)
1. Create Firebase project
2. Download service account key
3. Upload to Drive as `firebase-key.json`
4. Uncomment Firebase code in notebook

### 3. Prepare Input Images

- Create folder in Drive: `MyDrive/furniture_images/`
- Upload furniture images (JPG, PNG, WEBP)
- Images should have clear subjects with good lighting
- Recommended: White or simple backgrounds

### 4. Run the Pipeline

1. Run all cells in order
2. Monitor progress in output
3. Check results in Drive folder

## Output Structure

```
philo-homes-3d/
├── models/
│   ├── chair_001/
│   │   ├── chair_001.glb          # 3D model
│   │   ├── chair_001_albedo.png   # Base color texture
│   │   ├── chair_001_normal.png   # Normal map
│   │   ├── chair_001_roughness.png # Roughness map
│   │   ├── chair_001_metallic.png # Metallic map
│   │   ├── chair_001_preview.png  # Preview image
│   │   └── chair_001_nobg.png     # Processed input image
│   ├── sofa_002/
│   │   ├── sofa_002.glb
│   │   ├── sofa_002_albedo.png
│   │   ├── sofa_002_normal.png
│   │   ├── sofa_002_roughness.png
│   │   ├── sofa_002_metallic.png
│   │   ├── sofa_002_preview.png
│   │   └── sofa_002_nobg.png
│   └── ...
├── catalog/
│   └── furniture_3d_catalog.json  # Web integration catalog with PBR info
└── logs/
    └── processing_log_*.json      # Processing results
```

## Web Integration

The pipeline generates a catalog JSON file that can be used in your web application:

```json
{
  "id": "chair_001",
  "name": "Chair 001",
  "original_image": "chair_001.jpg",
  "processed_image": "chair_001_nobg.png",
  "model_url": "https://storage.googleapis.com/.../chair_001.glb",
  "assets": {
    "model": "https://storage.googleapis.com/.../chair_001.glb",
    "albedo": "https://storage.googleapis.com/.../chair_001_albedo.png",
    "normal": "https://storage.googleapis.com/.../chair_001_normal.png",
    "roughness": "https://storage.googleapis.com/.../chair_001_roughness.png",
    "metallic": "https://storage.googleapis.com/.../chair_001_metallic.png",
    "preview": "https://storage.googleapis.com/.../chair_001_preview.png",
    "processed_image": "https://storage.googleapis.com/.../chair_001_nobg.png"
  },
  "processed_date": "2024-01-20T10:30:00",
  "format": "glb",
  "has_pbr": true
}
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use smaller images (resize to 1024x1024)
   - Restart runtime

2. **Model Generation Fails**
   - Ensure image has clear subject
   - Check if background removal worked
   - Try different camera angles

3. **Slow Processing**
   - Normal: ~2-5 minutes per image
   - Use GPU runtime
   - Process in smaller batches

### Performance Tips

- Batch size: 10-20 images at a time
- Image size: 1024x1024 optimal
- Use consistent lighting in photos
- Avoid reflective surfaces

## API Integration

To integrate with your application:

```python
import requests
import json

# Fetch catalog
catalog_url = "your-storage-url/catalog/furniture_3d_catalog.json"
catalog = requests.get(catalog_url).json()

# Load 3D model in Three.js
for item in catalog:
    model_url = item['model_url']
    # Use GLTFLoader to load the model
```

## Advanced Configuration

### Hunyuan3D-2.1 Parameters

Modify in `generate_3d_model()`:
- `--guidance_scale`: Generation guidance (default: 7.5)
- `--num_inference_steps`: Quality steps (default: 30)
- `--do_texture_mapping`: Enable texture generation
- `--pbr_material`: Generate PBR materials
- `--fast_mode`: Quick generation without PBR

### Using Different Hunyuan3D Versions

```python
# Use Hunyuan3D-2.1 with PBR (highest quality)
pipeline = FurniturePipeline(input_dir, output_dir, use_pbr=True)

# Use fast mode without PBR (faster processing)
pipeline = FurniturePipeline(input_dir, output_dir, use_pbr=False)

# Use Hugging Face API (no local GPU needed)
# Replace generate_3d_model with generate_3d_model_hf_api in the pipeline
```

### Batch Processing Options

```python
# Process specific categories
image_files = [f for f in image_files if 'chair' in f.lower()]

# Process in chunks
chunk_size = 10
for i in range(0, len(image_files), chunk_size):
    chunk = image_files[i:i+chunk_size]
    results = pipeline.process_batch(chunk)
```

## Cost Estimation

- Google Colab Pro: ~$10/month
- Processing time: ~3 min/image with GPU
- Storage: ~5-10MB per 3D model
- Batch of 100 images: ~5 hours, ~1GB storage