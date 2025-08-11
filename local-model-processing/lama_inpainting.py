#!/usr/bin/env python3
"""
LaMa Inpainting for Object Removal
High-quality inpainting using LaMa (Large Mask Inpainting) model
"""

import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os
import sys
import zipfile
from datetime import datetime
from dataclasses import dataclass
import yaml
from omegaconf import OmegaConf

# Add LaMa model path
if os.path.exists("models/saicinpainting"):
    sys.path.insert(0, "models")

try:
    from saicinpainting.training.trainers import load_checkpoint
    from saicinpainting.evaluation.data import pad_tensor_to_modulo
    from saicinpainting.evaluation.refinement import refine_predict
    from saicinpainting.evaluation.utils import move_to_device
    LAMA_AVAILABLE = True
    REFINEMENT_AVAILABLE = True
    print("✅ LaMa modules loaded successfully!")
except ImportError as e:
    print(f"❌ LaMa modules not available: {e}")
    print("Please ensure LaMa is properly installed in models/saicinpainting/")
    LAMA_AVAILABLE = False
    REFINEMENT_AVAILABLE = False

@dataclass
class LamaConfig:
    use_official_refinement: bool = True
    refine_iterations: int = 1
    refine_mask_dilate: int = 15
    refine_mask_blur: int = 21
    hd_strategy_resize_limit: int = 2048
    hd_strategy_crop_margin: int = 196
    hd_strategy_crop_trigger_size: int = 1024
    refine_gpu_ids: str = "0"
    refine_n_iters: int = 15
    refine_lr: float = 0.002
    refine_min_side: int = 512
    refine_max_scales: int = 3
    refine_px_budget: int = 1800000

def get_best_device():
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        try:
            import torch_directml
            dml_device = torch_directml.device()
            print("✅ DirectML detected - Using AMD GPU acceleration!")
            return dml_device
        except:
            return torch.device('cpu')

device = get_best_device()
lama_device = 'cpu' if str(device).startswith('privateuseone') else device

current_state = {
    'image': None,
    'mask': None,
    'model': None,
    'config': None
}

def load_lama_model(checkpoint_path="models/best.ckpt", config_path="models/config.yaml"):
    """Load LaMa model"""
    if not LAMA_AVAILABLE:
        raise RuntimeError("LaMa modules not available")
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"LaMa checkpoint not found at {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"LaMa config not found at {config_path}")
        
    # Load config
    with open(config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    
    # Load model 
    map_loc = 'cpu' if str(device).startswith('privateuseone') else device
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=map_loc)
    model.freeze()
    
    try:
        model.to(lama_device)
        print(f"✅ LaMa model loaded on {lama_device}")
    except Exception as e:
        print(f"Warning: Could not move model to {lama_device}, using CPU")
        model.to('cpu')
    
    return model, train_config

def lama_inpaint(model, image: np.ndarray, mask: np.ndarray, config: LamaConfig) -> np.ndarray:
    """LaMa inpainting with proper refinement"""
    if model is None:
        raise ValueError("LaMa model not loaded")
    
    model_device = next(model.parameters()).device
    
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Prepare tensors
    image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    # Pad to multiple of 8
    image_tensor, mask_tensor = pad_tensor_to_modulo(image_tensor, 8), pad_tensor_to_modulo(mask_tensor, 8)
    
    # Move to device
    image_tensor = image_tensor.to(model_device)
    mask_tensor = mask_tensor.to(model_device)
    
    # Inpainting
    with torch.no_grad():
        batch = {'image': image_tensor, 'mask': mask_tensor}
        output = model(batch)
        
        if 'predicted_image' in output:
            inpainted = output['predicted_image']
        elif 'inpainted' in output:
            inpainted = output['inpainted']
        else:
            raise ValueError("No suitable output found in model")
    
    # Convert back
    inpainted = inpainted.cpu()
    result = inpainted[0].permute(1, 2, 0).numpy()
    
    # Handle value ranges
    if result.max() <= 1.0:
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
    else:
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Convert back to BGR if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # Unpad
    result = result[:image.shape[0], :image.shape[1]]
    
    return result

def lama_inpaint_with_refinement(model, image: np.ndarray, mask: np.ndarray, config: LamaConfig) -> np.ndarray:
    """LaMa inpainting with official multi-scale refinement"""
    if not REFINEMENT_AVAILABLE:
        print("⚠️ Official refinement not available, using single pass")
        return lama_inpaint(model, image, mask, config)
    
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to tensors
    image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    # Store original size
    orig_height, orig_width = image.shape[:2]
    
    # Pad to multiple of 8
    image_tensor = pad_tensor_to_modulo(image_tensor, 8)
    mask_tensor = pad_tensor_to_modulo(mask_tensor, 8)
    
    # Create batch dict
    batch = {
        'image': image_tensor,
        'mask': mask_tensor,
        'unpad_to_size': torch.tensor([orig_height, orig_width]).unsqueeze(0)
    }
    
    # For AMD GPU, refinement requires CUDA
    if str(device).startswith('privateuseone'):
        print("⚠️ AMD GPU detected - Official refinement requires CUDA, using single pass")
        return lama_inpaint(model, image, mask, config)
    
    try:
        print(f"Running official refinement with {config.refine_max_scales} scales...")
        result_tensor = refine_predict(
            batch, 
            model,
            gpu_ids=config.refine_gpu_ids,
            modulo=8,
            n_iters=config.refine_n_iters,
            lr=config.refine_lr,
            min_side=config.refine_min_side,
            max_scales=config.refine_max_scales,
            px_budget=config.refine_px_budget
        )
        result = result_tensor[0].permute(1, 2, 0).numpy()
        
        if result.max() <= 1.0:
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
        else:
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Convert back to BGR if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        print("✅ Official refinement complete!")
        return result
        
    except Exception as e:
        print(f"❌ Official refinement failed: {e}")
        print("Falling back to single pass...")
        return lama_inpaint(model, image, mask, config)

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
        
        current_state['image'] = image
        current_state['mask'] = mask
        
        # Clean up
        import shutil
        shutil.rmtree(extract_dir)
        
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), Image.fromarray(mask), "✅ Files loaded successfully"
        
    except Exception as e:
        return None, None, f"Error loading zip file: {str(e)}"

def process_inpainting(lama_config: LamaConfig):
    """Process inpainting with LaMa"""
    if current_state['image'] is None or current_state['mask'] is None:
        return None, None, "Please load image and mask first"
    
    img = current_state['image']
    mask = current_state['mask']
    
    # Apply mask preprocessing
    if lama_config.refine_mask_dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (lama_config.refine_mask_dilate * 2 + 1, lama_config.refine_mask_dilate * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)
    
    if lama_config.refine_mask_blur > 0:
        mask = cv2.GaussianBlur(mask, (lama_config.refine_mask_blur, lama_config.refine_mask_blur), 0)
    
    # Load LaMa model if not already loaded
    if current_state['model'] is None and LAMA_AVAILABLE:
        try:
            current_state['model'], current_state['config'] = load_lama_model()
        except Exception as e:
            return None, None, f"Failed to load LaMa model: {str(e)}"
    
    if current_state['model'] is None:
        return None, None, "LaMa model not available"
    
    try:
        if lama_config.use_official_refinement and REFINEMENT_AVAILABLE:
            result = lama_inpaint_with_refinement(current_state['model'], img, mask, lama_config)
            method = "LaMa with Official Multi-Scale Refinement"
        else:
            result = lama_inpaint(current_state['model'], img, mask, lama_config)
            method = "LaMa Single Pass"
        
        # Convert for display
        original_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        
        return original_pil, result_pil, f"✅ Inpainting complete using {method}"
        
    except Exception as e:
        return None, None, f"Inpainting failed: {str(e)}"

def save_result():
    """Save the current result"""
    if current_state.get('result') is None:
        return "No result to save"
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = f"results/lama_result_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        
        cv2.imwrite(result_path, current_state['result'])
        return f"✅ Result saved to: {result_path}"
        
    except Exception as e:
        return f"❌ Save failed: {str(e)}"

def create_ui():
    """Create Gradio UI for LaMa inpainting"""
    
    with gr.Blocks(
        title="LaMa Object Removal",
        theme=gr.themes.Base(),
        css=".container { max-width: 1400px; margin: auto; }"
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎨 LaMa Object Removal</h1>
            <p>High-quality inpainting using LaMa (Large Mask Inpainting) model</p>
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
                    gr.Markdown("### LaMa Parameters")
                    use_refinement = gr.Checkbox(
                        label="Use Official Multi-Scale Refinement",
                        value=True,
                        info="Recommended for best quality (requires CUDA)"
                    )
                    mask_dilate = gr.Slider(
                        minimum=0, maximum=50, value=15, step=5,
                        label="Mask Dilation"
                    )
                    mask_blur = gr.Slider(
                        minimum=0, maximum=51, value=21, step=2,
                        label="Mask Edge Blur"
                    )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        refine_scales = gr.Slider(
                            minimum=1, maximum=5, value=3, step=1,
                            label="Refinement Scales"
                        )
                        refine_iters = gr.Slider(
                            minimum=5, maximum=30, value=15, step=5,
                            label="Iterations per Scale"
                        )
                        refine_lr = gr.Slider(
                            minimum=0.001, maximum=0.01, value=0.002, step=0.001,
                            label="Learning Rate"
                        )
                
                process_btn = gr.Button("🎨 Remove Objects", variant="primary", size="lg")
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
        
        def on_process(use_ref, dilate, blur, scales, iters, lr):
            # Ensure blur is odd
            blur = blur if blur % 2 == 1 else blur + 1
            
            config = LamaConfig(
                use_official_refinement=use_ref,
                refine_mask_dilate=dilate,
                refine_mask_blur=blur,
                refine_max_scales=scales,
                refine_n_iters=iters,
                refine_lr=lr
            )
            return process_inpainting(config)
        
        zip_upload.upload(
            on_upload,
            inputs=[zip_upload],
            outputs=[input_image, input_mask, status]
        )
        
        process_btn.click(
            on_process,
            inputs=[use_refinement, mask_dilate, mask_blur, refine_scales, refine_iters, refine_lr],
            outputs=[original_display, result_display, status]
        )
        
        save_btn.click(save_result, outputs=[status])
    
    return demo

def check_models():
    """Check LaMa model availability"""
    checkpoint_path = "models/best.ckpt"
    config_path = "models/config.yaml"
    saicinpainting_path = "models/saicinpainting"
    
    print("\n📦 LaMa Model Status:")
    missing = []
    
    if os.path.exists(checkpoint_path):
        size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"  ✅ best.ckpt ({size:.1f} MB)")
    else:
        print(f"  ❌ best.ckpt")
        missing.append("best.ckpt")
    
    if os.path.exists(config_path):
        print(f"  ✅ config.yaml")
    else:
        print(f"  ❌ config.yaml")
        missing.append("config.yaml")
    
    if os.path.exists(saicinpainting_path):
        print(f"  ✅ saicinpainting/")
    else:
        print(f"  ❌ saicinpainting/")
        missing.append("saicinpainting/")
    
    if missing:
        print(f"\n⚠️ Missing {len(missing)} model files!")
        print("Run: python setup.py")
    
    return len(missing) == 0

if __name__ == "__main__":
    print("🚀 Starting LaMa Object Removal")
    print(f"📍 Device: {device}")
    print(f"🧠 LaMa Available: {LAMA_AVAILABLE}")
    print(f"🔄 Refinement Available: {REFINEMENT_AVAILABLE}")
    
    if str(device).startswith('privateuseone'):
        print("\n💡 AMD GPU detected - For best refinement:")
        print("   • Official refinement requires NVIDIA CUDA")
        print("   • Single-pass mode will be used instead")
    
    check_models()
    
    if not LAMA_AVAILABLE:
        print("\n❌ LaMa not available! Please run setup.py first.")
        exit(1)
    
    demo = create_ui()
    demo.launch(share=False, inbrowser=True)