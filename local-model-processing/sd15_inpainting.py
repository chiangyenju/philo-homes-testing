#!/usr/bin/env python3
"""
Stable Diffusion 1.5 Inpainting for Object Removal
Fast inpainting using Stable Diffusion 1.5 Inpainting model
"""

import gradio as gr
import cv2
import numpy as np
import torch
from PIL import Image
import os
import zipfile
from datetime import datetime
from dataclasses import dataclass

try:
    from diffusers import StableDiffusionInpaintPipeline
    DIFFUSERS_AVAILABLE = True
    print("✅ Diffusers modules loaded successfully!")
except ImportError as e:
    print(f"❌ Diffusers not available: {e}")
    print("Install with: pip install diffusers transformers accelerate")
    DIFFUSERS_AVAILABLE = False

@dataclass
class SD15Config:
    model_path: str = "models/sd15_inpaint"
    prompt: str = "empty room, clean walls, professional photography, architectural interior"
    negative_prompt: str = "furniture, people, objects, low quality, blurry, artifacts"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 1.0
    seed: int = -1

def get_best_device():
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_best_device()
current_state = {
    'image': None,
    'mask': None,
    'pipeline': None,
    'result': None
}

def load_sd15_pipeline(model_path):
    """Load SD 1.5 inpainting pipeline"""
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("Diffusers not available")
    
    # Check if local model exists
    if os.path.exists(model_path):
        print(f"Loading SD 1.5 from local path: {model_path}")
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            local_files_only=True
        )
    else:
        print("Downloading SD 1.5 inpainting model...")
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
    
    if device.type == 'cuda':
        pipeline = pipeline.to(device)
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except:
            pass  # XFormers not available
        pipeline.enable_attention_slicing()
    
    print(f"✅ SD 1.5 pipeline loaded on {device}")
    return pipeline

def load_zip_file(zip_file):
    """Load image and mask from uploaded zip file"""
    if zip_file is None:
        return None, None, "Please upload a zip file"
    
    try:
        # Extract zip file
        extract_dir = f"temp_extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Load image and mask
        image_path = os.path.join(extract_dir, "image.png")
        mask_path = os.path.join(extract_dir, "mask.png")
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            return None, None, "Zip file must contain image.png and mask.png"
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            return None, None, "Failed to load image or mask from zip file"
        
        # Convert to PIL Images
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        
        current_state['image'] = image_pil
        current_state['mask'] = mask_pil
        
        # Clean up
        import shutil
        shutil.rmtree(extract_dir)
        
        return image_pil, mask_pil, "✅ Files loaded successfully"
        
    except Exception as e:
        return None, None, f"Error loading zip file: {str(e)}"

def process_inpainting(config: SD15Config):
    """Process inpainting with SD 1.5"""
    if current_state['image'] is None or current_state['mask'] is None:
        return None, None, "Please load image and mask first"
    
    if not DIFFUSERS_AVAILABLE:
        return None, None, "Diffusers not available. Install with: pip install diffusers transformers accelerate"
    
    # Load pipeline if not already loaded
    if current_state['pipeline'] is None:
        try:
            current_state['pipeline'] = load_sd15_pipeline(config.model_path)
        except Exception as e:
            return None, None, f"Failed to load SD 1.5 pipeline: {str(e)}"
    
    try:
        print("Running SD 1.5 inpainting...")
        
        # Set seed if specified
        generator = None
        if config.seed != -1:
            generator = torch.Generator(device=device).manual_seed(config.seed)
        
        # Resize image if too large (SD 1.5 works best at 512x512)
        original_size = current_state['image'].size
        max_size = 512
        
        if max(original_size) > max_size:
            # Calculate new size maintaining aspect ratio
            scale = max_size / max(original_size)
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            
            # Resize both image and mask
            image_resized = current_state['image'].resize(new_size, Image.LANCZOS)
            mask_resized = current_state['mask'].resize(new_size, Image.NEAREST)
            
            print(f"Resized from {original_size} to {new_size} for optimal SD 1.5 performance")
        else:
            image_resized = current_state['image']
            mask_resized = current_state['mask']
        
        # Run inpainting
        result = current_state['pipeline'](
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            strength=config.strength,
            generator=generator,
            height=image_resized.height,
            width=image_resized.width
        ).images[0]
        
        # Resize back to original size if needed
        if max(original_size) > max_size:
            result = result.resize(original_size, Image.LANCZOS)
        
        current_state['result'] = result
        
        return current_state['image'], result, "✅ SD 1.5 inpainting complete!"
        
    except Exception as e:
        return None, None, f"Inpainting failed: {str(e)}"

def save_result():
    """Save the current result"""
    if current_state.get('result') is None:
        return "No result to save"
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = f"results/sd15_result_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        
        current_state['result'].save(result_path)
        return f"✅ Result saved to: {result_path}"
        
    except Exception as e:
        return f"❌ Save failed: {str(e)}"

def create_ui():
    """Create Gradio UI for SD 1.5 inpainting"""
    
    with gr.Blocks(
        title="SD 1.5 Object Removal",
        theme=gr.themes.Base(),
        css=".container { max-width: 1400px; margin: auto; }"
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>⚡ SD 1.5 Object Removal</h1>
            <p>Fast inpainting using Stable Diffusion 1.5 Inpainting model</p>
            <p><em>Upload the zip file generated by room_removal_ultimate.py</em></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                zip_upload = gr.File(
                    label="Upload Zip File",
                    file_types=[".zip"],
                    info="Upload the zip file from mask generation step"
                )
                
                with gr.Group():
                    gr.Markdown("### SD 1.5 Parameters")
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="empty room, clean walls, professional photography, architectural interior",
                        lines=3,
                        info="Describe what should fill the removed area"
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="furniture, people, objects, low quality, blurry, artifacts",
                        lines=2,
                        info="Describe what to avoid"
                    )
                    steps = gr.Slider(
                        minimum=20, maximum=100, value=50, step=5,
                        label="Inference Steps",
                        info="More steps = better quality but slower"
                    )
                    guidance = gr.Slider(
                        minimum=1.0, maximum=15.0, value=7.5, step=0.5,
                        label="Guidance Scale",
                        info="How closely to follow the prompt"
                    )
                    strength = gr.Slider(
                        minimum=0.1, maximum=1.0, value=1.0, step=0.01,
                        label="Strength",
                        info="How much to change the masked area"
                    )
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0
                    )
                
                process_btn = gr.Button("⚡ Remove Objects", variant="primary", size="lg")
                save_btn = gr.Button("💾 Save Result", variant="secondary", size="lg")
                
                status = gr.Textbox(label="Status", lines=3)
            
            with gr.Column(scale=2):
                with gr.Row():
                    input_image = gr.Image(label="Original Image", interactive=False)
                    input_mask = gr.Image(label="Mask", interactive=False)
                
                with gr.Row():
                    original_display = gr.Image(label="Before", interactive=False)
                    result_display = gr.Image(label="After", interactive=False)
        
        # Event handlers
        def on_upload(zip_file):
            return load_zip_file(zip_file)
        
        def on_process(prompt_text, neg_prompt_text, num_steps, guide_scale, strength_val, seed_val):
            config = SD15Config(
                prompt=prompt_text,
                negative_prompt=neg_prompt_text,
                num_inference_steps=num_steps,
                guidance_scale=guide_scale,
                strength=strength_val,
                seed=int(seed_val)
            )
            return process_inpainting(config)
        
        zip_upload.upload(
            on_upload,
            inputs=[zip_upload],
            outputs=[input_image, input_mask, status]
        )
        
        process_btn.click(
            on_process,
            inputs=[prompt, negative_prompt, steps, guidance, strength, seed],
            outputs=[original_display, result_display, status]
        )
        
        save_btn.click(save_result, outputs=[status])
    
    return demo

def check_models():
    """Check SD 1.5 model availability"""
    model_path = "models/sd15_inpaint"
    
    print("\n📦 SD 1.5 Model Status:")
    
    if os.path.exists(model_path):
        print(f"  ✅ Local SD 1.5 model found at: {model_path}")
        return True
    else:
        print(f"  ⚠️ Local SD 1.5 model not found at: {model_path}")
        print("  📥 Will download from HuggingFace on first use")
        return False

if __name__ == "__main__":
    print("🚀 Starting SD 1.5 Object Removal")
    print(f"📍 Device: {device}")
    print(f"🧠 Diffusers Available: {DIFFUSERS_AVAILABLE}")
    
    if device.type == 'cuda':
        print("✅ CUDA detected - will use GPU acceleration")
    else:
        print("⚠️ CUDA not available - using CPU (slow)")
    
    check_models()
    
    if not DIFFUSERS_AVAILABLE:
        print("\n❌ Diffusers not available!")
        print("Install with: pip install diffusers transformers accelerate")
        exit(1)
    
    demo = create_ui()
    demo.launch(share=False, inbrowser=True)