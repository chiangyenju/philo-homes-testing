#!/usr/bin/env python3
"""
Ultimate Room Object Removal - All-in-One Solution
Proper Big-LaMa implementation with full parameter control
"""

import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import sys
from typing import Dict, List, Tuple, Optional
import yaml
from omegaconf import OmegaConf
from dataclasses import dataclass

# Add paths for LaMa if available
if os.path.exists("models/saicinpainting"):
    sys.path.insert(0, "models")
if os.path.exists("lama"):
    sys.path.insert(0, "lama")

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Import export functionality for Colab workflow
try:
    from export_for_colab import export_for_colab
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False
    print("Export functionality not available - export_for_colab.py not found")

# Fix albumentations compatibility
try:
    import albumentations
    # Create compatibility wrapper for missing transforms
    if not hasattr(albumentations, 'DualIAATransform'):
        class DualIAATransform:
            pass
        albumentations.DualIAATransform = DualIAATransform
    if not hasattr(albumentations, 'IAATransform'):
        class IAATransform:
            pass
        albumentations.IAATransform = IAATransform
except:
    pass

# Try to import LaMa components
LAMA_AVAILABLE = False
REFINEMENT_AVAILABLE = False
try:
    from saicinpainting.training.trainers import load_checkpoint
    from saicinpainting.evaluation.data import pad_tensor_to_modulo
    LAMA_AVAILABLE = True
    print("✅ LaMa modules loaded successfully!")
    try:
        from saicinpainting.evaluation.refinement import refine_predict
        from saicinpainting.evaluation.utils import move_to_device
        REFINEMENT_AVAILABLE = True
        print("✅ LaMa refinement module loaded successfully!")
    except ImportError:
        print("⚠️ LaMa refinement module not available")
except ImportError as e:
    print(f"⚠️ LaMa modules not available: {e}")


@dataclass
class YOLOConfig:
    model: str = "yolov8x.pt"  # Use largest model for best detection
    confidence: float = 0.001  # Very low threshold as requested
    iou_threshold: float = 0.45
    max_detections: int = 300
    agnostic_nms: bool = False
    
@dataclass 
class SAMConfig:
    model_type: str = "vit_h"  # vit_b, vit_l, vit_h
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 100
    use_automatic_mask: bool = False
    multimask_output: bool = True
    mask_dilate_kernel: int = 0  # Dilate mask after creation
    mask_dilate_iterations: int = 1  # Number of dilation iterations
    
@dataclass
class LamaConfig:
    use_official_refinement: bool = True  # Use official multi-scale refinement
    refine_iterations: int = 5  # For old method - number of iterations
    refine_mask_dilate: int = 15
    refine_mask_blur: int = 21
    hd_strategy_resize_limit: int = 2048
    hd_strategy_crop_margin: int = 196
    hd_strategy_crop_trigger_size: int = 1024
    device: str = "cuda"
    # Official refinement parameters
    refine_gpu_ids: str = "0"  # GPU IDs for refinement
    refine_n_iters: int = 15  # Iterations per scale
    refine_lr: float = 0.002  # Learning rate
    refine_min_side: int = 512  # Min side for pyramid
    refine_max_scales: int = 3  # Max pyramid scales
    refine_px_budget: int = 1800000  # Max pixels to process
    
    
# Global state - Auto-detect best device
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
models = {}

# Device configuration for specific models
DEVICE_CONFIG = {
    'yolo': device,  # YOLO works with DirectML
    'sam': 'cpu' if str(device).startswith('privateuseone') else device,  # SAM needs CPU with DirectML
    'lama': 'cpu' if str(device).startswith('privateuseone') else device,  # LaMa needs CPU with DirectML
}
current_state = {
    'image': None,
    'detections': [],
    'masks': [],
    'combined_mask': None
}


def load_lama_model(checkpoint_path="models/best.ckpt", config_path="models/config.yaml"):
    """Load the real LaMa model properly"""
    if not LAMA_AVAILABLE:
        raise RuntimeError("LaMa modules not available")
        
    # Check files exist
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"LaMa checkpoint not found at {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"LaMa config not found at {config_path}")
        
    # Load config
    with open(config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    
    # For DirectML/CPU compatibility
    if str(device).startswith('privateuseone'):
        map_loc = 'cpu'
    else:
        map_loc = device
    
    # Load model 
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=map_loc)
    model.freeze()
    
    # Use configured device for LaMa
    lama_device = DEVICE_CONFIG['lama']
    if lama_device == 'cpu' and str(device).startswith('privateuseone'):
        print("  Using CPU for LaMa (DirectML doesn't support ComplexFloat)")
    
    try:
        model.to(lama_device)
    except Exception as e:
        print(f"Warning: Could not move model to {lama_device}, using CPU")
        model.to('cpu')
    
    return model, train_config


def lama_inpaint(model, image: np.ndarray, mask: np.ndarray, config: LamaConfig) -> np.ndarray:
    """
    LaMa inpainting with proper refinement (based on PR #112)
    First pass: Normal inpainting
    Refinement passes: Improve the inpainted areas
    """
    if model is None:
        raise ValueError("LaMa model not loaded")
    
    # Determine model device
    model_device = next(model.parameters()).device
    
    # Store original
    original_image = image.copy()
    
    # First, do standard inpainting (iteration 0)
    print("  Initial inpainting pass...")
    initial_result = None
    
    # Convert BGR to RGB if needed
    if image.shape[2] == 3:
        image_for_lama = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_for_lama = image
    
    # Prepare tensors
    image_tensor = torch.from_numpy(image_for_lama.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    # Pad to multiple of 8
    image_tensor, mask_tensor = pad_tensor_to_modulo(image_tensor, 8), pad_tensor_to_modulo(mask_tensor, 8)
    
    # Move to device
    image_tensor = image_tensor.to(model_device)
    mask_tensor = mask_tensor.to(model_device)
    
    # Initial inpainting
    with torch.no_grad():
        batch = {'image': image_tensor, 'mask': mask_tensor}
        output = model(batch)
        
        # Get the inpainted result
        if 'predicted_image' in output:
            inpainted = output['predicted_image']
            print("  Using 'predicted_image' output")
        elif 'inpainted' in output:
            inpainted = output['inpainted']
            print("  Using 'inpainted' output")
        else:
            raise ValueError("No suitable output found in model")
    
    # Convert back
    inpainted = inpainted.cpu()
    initial_result = inpainted[0].permute(1, 2, 0).numpy()
    
    # Handle value ranges
    if initial_result.max() <= 1.0:
        initial_result = np.clip(initial_result * 255, 0, 255).astype(np.uint8)
    else:
        initial_result = np.clip(initial_result, 0, 255).astype(np.uint8)
    
    # Convert back to BGR if needed
    if image.shape[2] == 3 and image_for_lama is not image:
        initial_result = cv2.cvtColor(initial_result, cv2.COLOR_RGB2BGR)
    
    # Unpad
    initial_result = initial_result[:image.shape[0], :image.shape[1]]
    
    # Check if we got a gray mask result
    mask_indices = np.where(mask > 0)
    if len(mask_indices[0]) > 100:
        sample_indices = np.random.choice(len(mask_indices[0]), min(100, len(mask_indices[0])), replace=False)
        gray_pixels = 0
        for idx in sample_indices:
            y, x = mask_indices[0][idx], mask_indices[1][idx]
            pixel = initial_result[y, x]
            if np.std(pixel) < 5 and 100 < np.mean(pixel) < 150:
                gray_pixels += 1
        
        if gray_pixels > 80:
            print("  ⚠️ Initial result has gray mask - skipping refinement")
            return initial_result
    
    # If requested, run refinement iterations
    result = initial_result.copy()
    
    if config.refine_iterations > 1:
        print(f"  Running {config.refine_iterations - 1} refinement iterations...")
        
        for iteration in range(1, config.refine_iterations):
            print(f"  Refinement iteration {iteration}/{config.refine_iterations - 1}")
            
            # For refinement, we work on the already inpainted image
            # The key insight from PR #112 is to use a slightly modified mask
            # that focuses on the boundaries and problem areas
            
            # Create refinement mask - slightly eroded version of original
            kernel_size = 2 * iteration + 3  # Gradually smaller masks
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            refine_mask = cv2.erode(mask, kernel, iterations=1)
            
            # Skip if mask becomes too small
            if np.sum(refine_mask) < 100:
                print(f"  Skipping iteration {iteration} - mask too small")
                continue
            
            # Use the current result as input for refinement
            if result.shape[2] == 3:
                refine_input = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            else:
                refine_input = result
            
            # Prepare tensors
            image_tensor = torch.from_numpy(refine_input.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            mask_tensor = torch.from_numpy(refine_mask).float().unsqueeze(0).unsqueeze(0) / 255.0
            
            # Pad
            image_tensor, mask_tensor = pad_tensor_to_modulo(image_tensor, 8), pad_tensor_to_modulo(mask_tensor, 8)
            
            # Move to device
            image_tensor = image_tensor.to(model_device)
            mask_tensor = mask_tensor.to(model_device)
            
            # Run refinement
            with torch.no_grad():
                batch = {'image': image_tensor, 'mask': mask_tensor}
                output = model(batch)
                
                if 'predicted_image' in output:
                    refined = output['predicted_image']
                elif 'inpainted' in output:
                    refined = output['inpainted']
                else:
                    continue
            
            # Convert back
            refined = refined.cpu()
            refined_result = refined[0].permute(1, 2, 0).numpy()
            
            if refined_result.max() <= 1.0:
                refined_result = np.clip(refined_result * 255, 0, 255).astype(np.uint8)
            else:
                refined_result = np.clip(refined_result, 0, 255).astype(np.uint8)
            
            # Convert back to BGR
            if result.shape[2] == 3 and refine_input is not result:
                refined_result = cv2.cvtColor(refined_result, cv2.COLOR_RGB2BGR)
            
            # Unpad
            refined_result = refined_result[:image.shape[0], :image.shape[1]]
            
            # Blend the refined areas back into the result
            # Only update the refined mask areas
            refine_mask_3ch = np.stack([refine_mask / 255.0] * 3, axis=-1)
            result = result * (1 - refine_mask_3ch) + refined_result * refine_mask_3ch
            result = result.astype(np.uint8)
    
    return result



def lama_inpaint_with_official_refinement(model, image: np.ndarray, mask: np.ndarray, config: LamaConfig) -> np.ndarray:
    """
    LaMa inpainting with official multi-scale refinement
    """
    if model is None:
        raise ValueError("LaMa model not loaded")
    
    if not REFINEMENT_AVAILABLE:
        print("⚠️ Official refinement not available, falling back to single pass")
        config_copy = LamaConfig(**vars(config))
        config_copy.refine_iterations = 1
        return lama_inpaint(model, image, mask, config_copy)
    
    # Convert BGR to RGB if needed
    if image.shape[2] == 3:
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
    
    # For AMD GPU/DirectML, we need to handle this differently
    if str(device).startswith('privateuseone'):
        print("  ⚠️ AMD GPU detected - Official refinement requires CUDA")
        print("  💡 Options:")
        print("     1. Use single-pass inpainting (disable refinement)")
        print("     2. Run this on Google Colab with GPU")
        print("     3. Use a cloud GPU service")
        # For now, fall back to single pass
        config_copy = LamaConfig(**vars(config))
        config_copy.refine_iterations = 1
        return lama_inpaint(model, image, mask, config_copy)
    
    gpu_ids_to_use = config.refine_gpu_ids
    model_was_on_gpu = False
    
    # Use official refinement
    print(f"  Running official refinement with {config.refine_max_scales} scales...")
    try:
        result_tensor = refine_predict(
            batch, 
            model,
            gpu_ids=gpu_ids_to_use,  # Use CPU for AMD
            modulo=8,
            n_iters=config.refine_n_iters,
            lr=config.refine_lr,
            min_side=config.refine_min_side,
            max_scales=config.refine_max_scales,
            px_budget=config.refine_px_budget
        )
        result = result_tensor[0].permute(1, 2, 0).numpy()
        
        # Move model back to original device if needed
        if model_was_on_gpu:
            model.to(DEVICE_CONFIG['lama'])
        
        # Result is already unpadded by refine_predict
        if result.max() <= 1.0:
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
        else:
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Convert back to BGR if needed
        if image.shape[2] == 3 and image_rgb is not image:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        print("  ✅ Official refinement complete!")
        return result
        
    except Exception as e:
        print(f"  ❌ Official refinement failed: {e}")
        print("  Falling back to single pass...")
        # Move model back if needed
        if 'model_was_on_gpu' in locals() and model_was_on_gpu:
            model.to(DEVICE_CONFIG['lama'])
        config_copy = LamaConfig(**vars(config))
        config_copy.refine_iterations = 1
        return lama_inpaint(model, image, mask, config_copy)


def process_detection(image, yolo_config: YOLOConfig):
    """Advanced YOLO detection with full parameter control"""
    global current_state
    
    if image is None:
        return None, "Please upload an image", ""
    
    current_state['image'] = np.array(image)
    img = current_state['image']
    
    # Load or update YOLO model
    if 'yolo' not in models or models['yolo'].model_name != yolo_config.model:
        # Always use models from the models/ folder
        local_model_path = f"models/{yolo_config.model}"
        if os.path.exists(local_model_path):
            print(f"Loading YOLO model from: {local_model_path}")
            models['yolo'] = YOLO(local_model_path)
        else:
            print(f"❌ YOLO model not found at {local_model_path}")
            print(f"Please run download_models.py to download the model")
            return None, "YOLO model not found", ""
    
    # Run detection with all parameters
    results = models['yolo'](
        img,
        conf=yolo_config.confidence,
        iou=yolo_config.iou_threshold,
        max_det=yolo_config.max_detections,
        agnostic_nms=yolo_config.agnostic_nms,
        verbose=False
    )
    
    # Process results
    current_state['detections'] = []
    detected_classes = []
    
    for r in results:
        for box in r.boxes:
            class_name = r.names[int(box.cls)]
            confidence = float(box.conf)
            bbox = box.xyxy[0].cpu().numpy()
            
            current_state['detections'].append({
                'bbox': bbox,
                'class': class_name,
                'confidence': confidence
            })
            detected_classes.append(f"{class_name} ({confidence:.3f})")
    
    # Visualize
    annotated = results[0].plot()
    
    info = f"Detected {len(current_state['detections'])} objects"
    details = "\n".join(detected_classes[:20])  # Show first 20
    if len(detected_classes) > 20:
        details += f"\n... and {len(detected_classes) - 20} more"
    
    return Image.fromarray(annotated), info, details


def process_segmentation(selected_indices, sam_config: SAMConfig):
    """Advanced SAM segmentation with full parameter control"""
    global current_state
    
    if current_state['image'] is None:
        return None, "Please run detection first"
    
    if not selected_indices:
        return None, "Please select objects to remove"
    
    # Parse selected indices
    try:
        if selected_indices.strip().lower() == 'all':
            # Select all detected objects
            indices = list(range(len(current_state['detections'])))
            if not indices:
                return None, "No objects detected yet"
        else:
            indices = [int(i.strip()) for i in selected_indices.split(',') if i.strip()]
    except:
        return None, "Invalid indices format. Use comma-separated numbers or 'all'."
    
    # Load SAM model
    model_map = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth", 
        "vit_h": "sam_vit_h_4b8939.pth"
    }
    
    checkpoint = f"models/{model_map.get(sam_config.model_type, 'sam_vit_b_01ec64.pth')}"
    
    if 'sam' not in models or models.get('sam_type') != sam_config.model_type:
        print(f"Loading SAM model: {sam_config.model_type}")
        if not os.path.exists(checkpoint):
            checkpoint = f"models/sam_vit_b_01ec64.pth"  # Fallback
        models['sam'] = sam_model_registry[sam_config.model_type](checkpoint=checkpoint)
        # Use configured device for SAM
        models['sam'].to(device=DEVICE_CONFIG['sam'])
        if DEVICE_CONFIG['sam'] == 'cpu' and str(device).startswith('privateuseone'):
            print("  Using CPU for SAM (DirectML compatibility)")
        models['sam_type'] = sam_config.model_type
    
    # Create masks
    img = current_state['image']
    h, w = img.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    successful_masks = 0
    failed_masks = []
    
    if sam_config.use_automatic_mask:
        # Automatic mask generation for selected objects
        mask_generator = SamAutomaticMaskGenerator(
            models['sam'],
            points_per_side=sam_config.points_per_side,
            pred_iou_thresh=sam_config.pred_iou_thresh,
            stability_score_thresh=sam_config.stability_score_thresh,
            crop_n_layers=sam_config.crop_n_layers,
            crop_n_points_downscale_factor=sam_config.crop_n_points_downscale_factor,
            min_mask_region_area=sam_config.min_mask_region_area,
        )
        auto_masks = mask_generator.generate(img)
        
        # Filter masks that overlap with selected detections
        for idx in indices:
            if 0 <= idx < len(current_state['detections']):
                det = current_state['detections'][idx]
                bbox = det['bbox'].astype(int)
                x1, y1, x2, y2 = bbox
                
                # Find masks that overlap with this detection
                for mask_data in auto_masks:
                    mask = mask_data['segmentation']
                    # Check if mask overlaps with bbox
                    bbox_mask = np.zeros_like(mask)
                    bbox_mask[y1:y2, x1:x2] = 1
                    overlap = np.sum(mask & bbox_mask)
                    if overlap > 0.5 * np.sum(bbox_mask):  # 50% overlap
                        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
                        successful_masks += 1
                        break
    else:
        # Prompt-based segmentation
        predictor = SamPredictor(models['sam'])
        predictor.set_image(img)
        
        for idx in indices:
            if 0 <= idx < len(current_state['detections']):
                det = current_state['detections'][idx]
                bbox = det['bbox'].astype(int)
                
                try:
                    # Convert bbox to SAM format (xyxy)
                    # Ensure coordinates are within image bounds
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    
                    # SAM expects numpy array format
                    input_box = np.array([x1, y1, x2, y2])
                    
                    # Use bounding box as prompt
                    masks, scores, logits = predictor.predict(
                        box=input_box,
                        multimask_output=sam_config.multimask_output
                    )
                    
                    # Select best mask
                    if sam_config.multimask_output and len(masks) > 0:
                        best_idx = np.argmax(scores)
                        mask = masks[best_idx]
                    else:
                        mask = masks[0] if len(masks) > 0 else None
                    
                    if mask is not None:
                        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
                        successful_masks += 1
                    else:
                        failed_masks.append(idx)
                        
                except Exception as e:
                    print(f"Failed to create mask for object {idx}: {e}")
                    failed_masks.append(idx)
    
    # Apply mask dilation if requested
    if sam_config.mask_dilate_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (sam_config.mask_dilate_kernel * 2 + 1, sam_config.mask_dilate_kernel * 2 + 1)
        )
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=sam_config.mask_dilate_iterations)
    
    current_state['combined_mask'] = combined_mask * 255
    
    # Visualize
    mask_viz = np.zeros_like(img)
    mask_viz[:,:,0] = current_state['combined_mask']  # Red channel
    overlay = cv2.addWeighted(img, 0.7, mask_viz, 0.3, 0)
    
    if 'successful_masks' in locals():
        mask_info = f"Created masks for {successful_masks}/{len(indices)} objects"
        if failed_masks:
            mask_info += f" (failed: {failed_masks})"
    else:
        mask_info = f"Created mask for {len(indices)} objects"
    
    if sam_config.mask_dilate_kernel > 0:
        mask_info += f" (dilated {sam_config.mask_dilate_kernel}px × {sam_config.mask_dilate_iterations})"
    
    return Image.fromarray(overlay), mask_info


def process_inpainting(lama_config: LamaConfig):
    """Advanced LaMa inpainting with full control"""
    global current_state
    
    if current_state['image'] is None or current_state['combined_mask'] is None:
        return None, "Please complete detection and segmentation first"
    
    img = current_state['image']
    mask = current_state['combined_mask']
    
    # Apply mask preprocessing
    if lama_config.refine_mask_dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (lama_config.refine_mask_dilate * 2 + 1, lama_config.refine_mask_dilate * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)
    
    # Load LaMa model if available
    if 'lama' not in models and LAMA_AVAILABLE:
        try:
            models['lama'], models['lama_config'] = load_lama_model()
            print("✅ LaMa model loaded!")
        except Exception as e:
            print(f"❌ Failed to load LaMa: {e}")
            import traceback
            traceback.print_exc()
            models['lama'] = None
    
    if models.get('lama') is not None:
        # Use real LaMa with proper refinement
        try:
            # Check if we're on AMD
            is_amd = str(device).startswith('privateuseone')
            
            if lama_config.use_official_refinement:
                if REFINEMENT_AVAILABLE and not is_amd:
                    print(f"Running LaMa with official multi-scale refinement...")
                    result = lama_inpaint_with_official_refinement(models['lama'], img, mask, lama_config)
                    method = "LaMa with Official Multi-Scale Refinement"
                elif is_amd:
                    print(f"\n⚠️ AMD GPU detected - Official refinement requires NVIDIA CUDA")
                    print(f"📋 For best quality, you have these options:")
                    print(f"   1. Use the included Colab notebook: lama_refinement_colab.ipynb")
                    print(f"   2. Save image/mask and process on a cloud GPU service")
                    print(f"   3. Use single-pass inpainting (good quality, no refinement)\n")
                    print(f"Using single-pass inpainting for now...")
                    lama_config_copy = LamaConfig(**vars(lama_config))
                    lama_config_copy.refine_iterations = 1
                    result = lama_inpaint(models['lama'], img, mask, lama_config_copy)
                    method = "LaMa Single Pass (AMD GPU - Refinement needs CUDA)"
                else:
                    print(f"⚠️ Refinement not available, using single pass...")
                    lama_config_copy = LamaConfig(**vars(lama_config))
                    lama_config_copy.refine_iterations = 1
                    result = lama_inpaint(models['lama'], img, mask, lama_config_copy)
                    method = "LaMa Single Pass (No refinement available)"
            else:
                # Use old iterative method or single pass
                if lama_config.refine_iterations > 1:
                    print(f"⚠️ Warning: Multiple iterations may create artifacts!")
                    print(f"Running LaMa inpainting with {lama_config.refine_iterations} iterations...")
                else:
                    print(f"Running LaMa inpainting (single pass)...")
                result = lama_inpaint(models['lama'], img, mask, lama_config)
                method = f"LaMa with {lama_config.refine_iterations} iteration(s)"
            print("✅ LaMa inpainting complete!")
        except Exception as e:
            print(f"❌ LaMa inference failed: {e}")
            import traceback
            traceback.print_exc()
            result = img  # Return original
            method = "Failed - LaMa error"
    else:
        # No LaMa available
        print("❌ LaMa model not loaded!")
        result = img  # Return original
        method = "Failed - No LaMa model"
    
    # Return both original and result for comparison
    original_pil = Image.fromarray(img)
    result_pil = Image.fromarray(result)
    
    return original_pil, result_pil, f"Inpainting complete using {method}"


def create_ui():
    """Create advanced Gradio UI with full parameter control"""
    
    with gr.Blocks(
        title="Ultimate Room Object Removal",
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="gray",
            spacing_size="md",
            radius_size="md",
        ),
        css="""
        .container { max-width: 1400px; margin: auto; }
        .parameter-box { background: #f0f0f0; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .detection-list { max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 12px; }
        """
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="font-size: 2.5em; margin-bottom: 10px;">🏠 Ultimate Room Object Removal</h1>
            <p style="font-size: 1.2em; color: #666;">Advanced inpainting with full parameter control</p>
        </div>
        """)
        
        with gr.Tabs():
            with gr.Tab("🎯 Object Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(label="Upload Image", type="pil")
                        
                        with gr.Group():
                            gr.Markdown("### YOLO Parameters")
                            yolo_model = gr.Dropdown(
                                choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                                value="yolov8x.pt",
                                label="Model Size"
                            )
                            yolo_conf = gr.Slider(
                                minimum=0.001, maximum=1.0, value=0.001, step=0.001,
                                label="Confidence Threshold",
                                info="Lower = detect more objects"
                            )
                            yolo_iou = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.45, step=0.05,
                                label="IoU Threshold",
                                info="Overlap threshold for Non-Maximum Suppression (0.45 = balanced, lower = more boxes)"
                            )
                            yolo_max_det = gr.Slider(
                                minimum=10, maximum=1000, value=300, step=10,
                                label="Max Detections",
                                info="Maximum number of objects to detect per image"
                            )
                            yolo_agnostic = gr.Checkbox(
                                label="Class-Agnostic NMS",
                                value=False,
                                info="Treat all classes the same during NMS (useful for general object removal)"
                            )
                        
                        detect_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        detection_output = gr.Image(label="Detection Results")
                        detection_info = gr.Textbox(label="Summary", lines=1)
                        detection_list = gr.Textbox(
                            label="Detected Objects (index: class confidence)",
                            lines=10,
                            elem_classes=["detection-list"]
                        )
            
            with gr.Tab("✂️ Segmentation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        selected_objects = gr.Textbox(
                            label="Objects to Remove",
                            placeholder="Enter indices (e.g., 0,2,5,10) or 'all'",
                            info="Comma-separated indices from detection list, or 'all' for everything"
                        )
                        
                        with gr.Row():
                            select_all_btn = gr.Button("Select All Objects", size="sm")
                            clear_selection_btn = gr.Button("Clear Selection", size="sm")
                        
                        with gr.Group():
                            gr.Markdown("### SAM Parameters")
                            sam_model = gr.Dropdown(
                                choices=["vit_b", "vit_l", "vit_h"],
                                value="vit_b",
                                label="Model Type",
                                info="Larger = better quality but slower"
                            )
                            sam_multimask = gr.Checkbox(
                                label="Multi-mask Output",
                                value=True,
                                info="Generate multiple masks and select best"
                            )
                            
                            with gr.Accordion("Advanced SAM Settings", open=False):
                                sam_auto = gr.Checkbox(
                                    label="Use Automatic Mask Generation",
                                    value=False,
                                    info="Generate masks automatically without prompts"
                                )
                                sam_points = gr.Slider(
                                    minimum=8, maximum=64, value=32, step=8,
                                    label="Points per Side (for auto)",
                                    visible=False,
                                    info="Grid density for automatic mask generation"
                                )
                                sam_pred_iou = gr.Slider(
                                    minimum=0.5, maximum=1.0, value=0.88, step=0.02,
                                    label="Predicted IoU Threshold",
                                    info="Higher = only keep high-confidence masks (0.88 recommended)"
                                )
                                sam_stability = gr.Slider(
                                    minimum=0.5, maximum=1.0, value=0.95, step=0.02,
                                    label="Stability Score Threshold",
                                    info="Higher = only keep stable masks across augmentations"
                                )
                                
                            with gr.Accordion("Mask Post-Processing", open=True):
                                sam_dilate_kernel = gr.Slider(
                                    minimum=0, maximum=50, value=10, step=5,
                                    label="Mask Dilation Kernel Size",
                                    info="Expand mask boundaries (0 = no dilation, 10-20 recommended for shadows)"
                                )
                                sam_dilate_iter = gr.Slider(
                                    minimum=1, maximum=5, value=1, step=1,
                                    label="Dilation Iterations",
                                    info="Number of times to apply dilation (1-2 usually sufficient)"
                                )
                                
                                # Show/hide auto parameters
                                sam_auto.change(
                                    lambda x: gr.update(visible=x),
                                    inputs=[sam_auto],
                                    outputs=[sam_points]
                                )
                        
                        segment_btn = gr.Button("✂️ Create Masks", variant="primary", size="lg")
                        
                        # Add export button for Colab workflow
                        with gr.Row(visible=False) as export_row:
                            export_btn = gr.Button("Export for Colab Processing", variant="secondary", size="lg")
                            export_info = gr.Textbox(label="Export Status", lines=1, visible=False)
                    
                    with gr.Column(scale=2):
                        mask_output = gr.Image(label="Segmentation Mask")
                        mask_info = gr.Textbox(label="Mask Info", lines=1)
            
            with gr.Tab("🎨 Inpainting"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### LaMa Parameters")
                            use_official_refine = gr.Checkbox(
                                label="Use Official Multi-Scale Refinement",
                                value=True,
                                info="Recommended: Uses proper pyramid refinement instead of iterative method"
                            )
                            lama_iterations = gr.Slider(
                                minimum=1, maximum=10, value=1, step=1,
                                label="Iterations (Old Method)",
                                info="Only used if official refinement is OFF. WARNING: >1 may cause artifacts!"
                            )
                            lama_dilate = gr.Slider(
                                minimum=0, maximum=50, value=15, step=5,
                                label="Mask Dilation",
                                info="Expand mask before inpainting"
                            )
                            lama_blur = gr.Slider(
                                minimum=0, maximum=51, value=21, step=2,
                                label="Mask Edge Blur",
                                info="Soften mask edges (must be odd)"
                            )
                            
                            with gr.Accordion("Official Refinement Settings", open=False):
                                refine_scales = gr.Slider(
                                    minimum=1, maximum=5, value=3, step=1,
                                    label="Pyramid Scales",
                                    info="Number of multi-scale levels (3 recommended)"
                                )
                                refine_iters = gr.Slider(
                                    minimum=5, maximum=30, value=15, step=5,
                                    label="Iterations per Scale",
                                    info="Optimization iterations at each scale"
                                )
                                refine_lr = gr.Slider(
                                    minimum=0.001, maximum=0.01, value=0.002, step=0.001,
                                    label="Learning Rate",
                                    info="Feature optimization learning rate"
                                )
                            
                            with gr.Accordion("HD Strategy", open=False):
                                lama_resize_limit = gr.Slider(
                                    minimum=512, maximum=4096, value=2048, step=256,
                                    label="Resize Limit"
                                )
                                lama_crop_margin = gr.Slider(
                                    minimum=0, maximum=512, value=196, step=32,
                                    label="Crop Margin"
                                )
                                lama_crop_size = gr.Slider(
                                    minimum=512, maximum=2048, value=1024, step=128,
                                    label="Crop Trigger Size"
                                )
                        
                        inpaint_btn = gr.Button("🎨 Remove Objects", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        original_display = gr.Image(label="Original Image", interactive=False)
                        inpaint_output = gr.Image(label="Final Result")
                        inpaint_info = gr.Textbox(label="Process Info", lines=1)
        
        with gr.Tab("📚 Parameter Guide"):
            gr.HTML("""
            <div style="padding: 20px;">
                <h2>🎯 YOLO Detection Parameters</h2>
                <ul style="line-height: 1.8;">
                    <li><b>Confidence Threshold:</b> Controls detection sensitivity (0.001 = detect everything, 0.5 = only confident detections)</li>
                    <li><b>IoU Threshold:</b> Controls box overlap for duplicate removal (0.45 standard, 0.3 = keep more overlapping boxes)</li>
                    <li><b>Max Detections:</b> Limits total objects detected (increase for cluttered scenes)</li>
                    <li><b>Class-Agnostic NMS:</b> Useful when removing any object type regardless of class</li>
                </ul>
                
                <h2>✂️ SAM Segmentation Parameters</h2>
                <ul style="line-height: 1.8;">
                    <li><b>Model Type:</b> vit_b = fast (358MB), vit_h = best quality (2.4GB)</li>
                    <li><b>Multi-mask Output:</b> Generate 3 masks and pick best (recommended ON)</li>
                    <li><b>Predicted IoU Threshold:</b> Confidence filter for masks (0.88 = balanced, 0.95 = very strict)</li>
                    <li><b>Stability Score:</b> How consistent mask is across augmentations (0.95 recommended)</li>
                    <li><b>Mask Dilation:</b> Expand mask to include shadows/edges (10-20 pixels typical)</li>
                    <li><b>Dilation Iterations:</b> How many times to apply dilation (1-2 usually enough)</li>
                </ul>
                
                <h2>🎨 LaMa Inpainting Parameters</h2>
                <ul style="line-height: 1.8;">
                    <li><b>Official Refinement:</b> Uses multi-scale pyramid optimization (RECOMMENDED)</li>
                    <li><b>Pyramid Scales:</b> Number of resolution levels (3 = good balance)</li>
                    <li><b>Iterations per Scale:</b> Optimization steps at each scale (15 recommended)</li>
                    <li><b>Learning Rate:</b> Feature optimization rate (0.002 is optimal)</li>
                    <li><b>Old Method Iterations:</b> Simple iterative approach (WARNING: >1 causes artifacts!)</li>
                    <li><b>Mask Dilation:</b> Extra expansion before inpainting (15 recommended)</li>
                    <li><b>Mask Edge Blur:</b> Soften mask boundaries for seamless blending (21 recommended)</li>
                    <li><b>HD Strategy:</b> For images >1024px - resize limit, crop margin, trigger size</li>
                </ul>
                
                <h2>🔥 Quick Settings</h2>
                <ul style="line-height: 1.8;">
                    <li><b>Maximum Detection:</b> Confidence=0.001, IoU=0.3, Max=500</li>
                    <li><b>Best Quality:</b> Official refinement ON, 3 scales, SAM dilation=15</li>
                    <li><b>Fast Mode:</b> Official refinement OFF, 1 iteration, vit_b model</li>
                    <li><b>Avoid Artifacts:</b> NEVER use old method with >1 iteration!</li>
                </ul>
            </div>
            """)
        
        gr.HTML("""
        <div style="margin-top: 40px; padding: 20px; background: #f9f9f9; border-radius: 10px;">
            <h3>💡 Tips for Best Results</h3>
            <ul>
                <li><b>Detection:</b> Use confidence 0.001-0.01 for maximum detection</li>
                <li><b>Segmentation:</b> vit_h gives best quality but is slower</li>
                <li><b>Inpainting:</b> 5-7 iterations with proper LaMa gives best results</li>
                <li><b>Mask:</b> Slight dilation (10-20) helps remove shadows</li>
                <li><b>AMD GPU Users:</b> After generating masks, use "Export for Colab" button to process with GPU acceleration</li>
                <li><b>Workflow:</b> You can run detection/segmentation locally and export to Colab for refinement</li>
            </ul>
        </div>
        """)
        
        # Event handlers
        def on_detect(img, model, conf, iou, max_det, agnostic):
            config = YOLOConfig(
                model=model,
                confidence=conf,
                iou_threshold=iou,
                max_detections=max_det,
                agnostic_nms=agnostic
            )
            return process_detection(img, config)
        
        def select_all_objects():
            if current_state['detections']:
                return "all"
            return ""
        
        def clear_objects():
            return ""
        
        def on_segment(indices, model, multimask, auto, points, pred_iou, stability, dilate_kernel, dilate_iter):
            config = SAMConfig(
                model_type=model,
                multimask_output=multimask,
                use_automatic_mask=auto,
                points_per_side=points,
                pred_iou_thresh=pred_iou,
                stability_score_thresh=stability,
                mask_dilate_kernel=dilate_kernel,
                mask_dilate_iterations=dilate_iter
            )
            return process_segmentation(indices, config)
        
        def on_inpaint(use_official, iterations, dilate, blur, scales, iters_per_scale, lr, resize_limit, crop_margin, crop_size):
            # Ensure blur is odd
            blur = blur if blur % 2 == 1 else blur + 1
            
            config = LamaConfig(
                use_official_refinement=use_official,
                refine_iterations=iterations,
                refine_mask_dilate=dilate,
                refine_mask_blur=blur,
                refine_max_scales=scales,
                refine_n_iters=iters_per_scale,
                refine_lr=lr,
                hd_strategy_resize_limit=resize_limit,
                hd_strategy_crop_margin=crop_margin,
                hd_strategy_crop_trigger_size=crop_size
            )
            return process_inpainting(config)
        
        detect_btn.click(
            on_detect,
            inputs=[input_image, yolo_model, yolo_conf, yolo_iou, yolo_max_det, yolo_agnostic],
            outputs=[detection_output, detection_info, detection_list]
        )
        
        # Selection button handlers
        select_all_btn.click(
            select_all_objects,
            outputs=[selected_objects]
        )
        
        clear_selection_btn.click(
            clear_objects,
            outputs=[selected_objects]
        )
        
        def on_segment_with_export(*args):
            """Wrapper to handle segmentation and show export button"""
            mask_img, info = on_segment(*args)
            # Show export button if mask was created successfully
            if mask_img is not None and current_state['combined_mask'] is not None:
                return mask_img, info, gr.update(visible=True), gr.update(visible=True)
            return mask_img, info, gr.update(visible=False), gr.update(visible=False)
        
        def on_export_for_colab():
            """Export current image and mask for Colab processing"""
            if not EXPORT_AVAILABLE:
                return "Export functionality not available - export_for_colab.py not found"
            
            if current_state['image'] is None or current_state['combined_mask'] is None:
                return "No image/mask to export. Please generate a mask first."
            
            try:
                export_path = export_for_colab(
                    current_state['image'],
                    current_state['combined_mask'],
                    create_zip=True
                )
                return f"[OK] Exported to: {export_path}.zip\n[INFO] Upload to Google Drive and run lama_refinement_colab.ipynb"
            except Exception as e:
                return f"[ERROR] Export failed: {str(e)}"
        
        segment_btn.click(
            on_segment_with_export,
            inputs=[selected_objects, sam_model, sam_multimask, sam_auto, 
                   sam_points, sam_pred_iou, sam_stability, sam_dilate_kernel, sam_dilate_iter],
            outputs=[mask_output, mask_info, export_row, export_info]
        )
        
        export_btn.click(
            on_export_for_colab,
            outputs=[export_info]
        )
        
        inpaint_btn.click(
            on_inpaint,
            inputs=[use_official_refine, lama_iterations, lama_dilate, lama_blur,
                   refine_scales, refine_iters, refine_lr,
                   lama_resize_limit, lama_crop_margin, lama_crop_size],
            outputs=[original_display, inpaint_output, inpaint_info]
        )
    
    return demo


def check_models():
    """Check which models are available locally"""
    models_status = {
        "yolov8x.pt": "YOLOv8x (Best detection)",
        "yolov8l.pt": "YOLOv8l (Large detection)",
        "yolov8m.pt": "YOLOv8m (Medium detection)",
        "sam_vit_h_4b8939.pth": "SAM ViT-H (Best segmentation)",
        "sam_vit_b_01ec64.pth": "SAM ViT-B (Fast segmentation)",
        "sam_vit_l_0b3195.pth": "SAM ViT-L (Balanced segmentation)",
        "best.ckpt": "Big-LaMa checkpoint",
        "config.yaml": "Big-LaMa config"
    }
    
    print("\n📦 Model Status:")
    missing = []
    for model_file, description in models_status.items():
        path = f"models/{model_file}"
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"  ✅ {model_file} ({size:.1f} MB) - {description}")
        else:
            print(f"  ❌ {model_file} - {description}")
            missing.append(model_file)
    
    if missing:
        print(f"\n⚠️ Missing {len(missing)} model files!")
        print("Run: python download_models.py")
        print("This will download all required models for offline use.\n")
    
    return len(missing) == 0


if __name__ == "__main__":
    print(f"🚀 Starting Ultimate Room Object Removal")
    print(f"📍 Device: {device}")
    print(f"🧠 LaMa Available: {LAMA_AVAILABLE}")
    print(f"🔄 Official Refinement Available: {REFINEMENT_AVAILABLE}")
    
    if str(device).startswith('privateuseone'):
        print(f"\n💡 AMD GPU Detected - For best refinement quality:")
        print(f"   • Use the included Google Colab notebook: lama_refinement_colab.ipynb")
        print(f"   • Or use single-pass mode (still produces good results)\n")
    
    # Check models
    all_models_available = check_models()
    
    if not LAMA_AVAILABLE:
        print("\n⚠️  For best results, set up LaMa:")
        print("1. git clone https://github.com/advimman/lama.git")
        print("2. Copy saicinpainting folder to models/")
        print("3. Restart the script")
    
    demo = create_ui()
    demo.launch(share=False, inbrowser=True)