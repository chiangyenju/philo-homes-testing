#!/usr/bin/env python3
"""
Room Object Detection and Mask Generation
Generates precise masks for object removal and exports for inpainting models
"""

import gradio as gr
import cv2
import numpy as np
import torch
from PIL import Image
import os
import sys
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

@dataclass
class YOLOConfig:
    model: str = "yolov8x.pt"
    confidence: float = 0.001
    iou_threshold: float = 0.45
    max_detections: int = 300
    agnostic_nms: bool = False

@dataclass 
class SAMConfig:
    model_type: str = "vit_h"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 100
    use_automatic_mask: bool = False
    multimask_output: bool = True
    mask_dilate_kernel: int = 10
    mask_dilate_iterations: int = 1

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
DEVICE_CONFIG = {
    'yolo': device,
    'sam': 'cpu' if str(device).startswith('privateuseone') else device,
}

current_state = {
    'image': None,
    'detections': [],
    'masks': [],
    'combined_mask': None
}

def process_detection(image, yolo_config: YOLOConfig):
    """YOLO object detection"""
    global current_state
    
    if image is None:
        return None, "Please upload an image", ""
    
    current_state['image'] = np.array(image)
    img = current_state['image']
    
    # Load YOLO model
    if 'yolo' not in models or models['yolo'].model_name != yolo_config.model:
        local_model_path = f"models/{yolo_config.model}"
        if os.path.exists(local_model_path):
            print(f"Loading YOLO model from: {local_model_path}")
            models['yolo'] = YOLO(local_model_path)
        else:
            return None, "YOLO model not found - run setup.py first", ""
    
    # Run detection
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
            detected_classes.append(f"[{len(current_state['detections'])-1}] {class_name} ({confidence:.3f})")
    
    annotated = results[0].plot()
    info = f"Detected {len(current_state['detections'])} objects"
    details = "\n".join(detected_classes[:20])
    if len(detected_classes) > 20:
        details += f"\n... and {len(detected_classes) - 20} more"
    
    return Image.fromarray(annotated), info, details

def process_segmentation(selected_indices, sam_config: SAMConfig):
    """SAM segmentation to create precise masks"""
    global current_state
    
    if current_state['image'] is None:
        return None, "Please run detection first"
    
    if not selected_indices:
        return None, "Please select objects to remove"
    
    # Parse indices
    try:
        if selected_indices.strip().lower() == 'all':
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
            return None, f"SAM model not found at {checkpoint} - run setup.py first"
        models['sam'] = sam_model_registry[sam_config.model_type](checkpoint=checkpoint)
        models['sam'].to(device=DEVICE_CONFIG['sam'])
        models['sam_type'] = sam_config.model_type
    
    # Create masks
    img = current_state['image']
    h, w = img.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    successful_masks = 0
    failed_masks = []
    
    predictor = SamPredictor(models['sam'])
    predictor.set_image(img)
    
    for idx in indices:
        if 0 <= idx < len(current_state['detections']):
            det = current_state['detections'][idx]
            bbox = det['bbox'].astype(int)
            
            try:
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                input_box = np.array([x1, y1, x2, y2])
                
                masks, scores, logits = predictor.predict(
                    box=input_box,
                    multimask_output=sam_config.multimask_output
                )
                
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
    
    # Apply mask dilation
    if sam_config.mask_dilate_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (sam_config.mask_dilate_kernel * 2 + 1, sam_config.mask_dilate_kernel * 2 + 1)
        )
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=sam_config.mask_dilate_iterations)
    
    current_state['combined_mask'] = combined_mask * 255
    
    # Visualize
    mask_viz = np.zeros_like(img)
    mask_viz[:,:,0] = current_state['combined_mask']
    overlay = cv2.addWeighted(img, 0.7, mask_viz, 0.3, 0)
    
    mask_info = f"Created masks for {successful_masks}/{len(indices)} objects"
    if failed_masks:
        mask_info += f" (failed: {failed_masks})"
    if sam_config.mask_dilate_kernel > 0:
        mask_info += f" (dilated {sam_config.mask_dilate_kernel}px × {sam_config.mask_dilate_iterations})"
    
    return Image.fromarray(overlay), mask_info

def export_for_inpainting():
    """Export image and mask as zip file for inpainting models"""
    if current_state['image'] is None or current_state['combined_mask'] is None:
        return "No image/mask to export. Please generate a mask first."
    
    try:
        # Create export directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = f"exports/{timestamp}"
        os.makedirs(export_dir, exist_ok=True)
        
        # Save image and mask
        img_pil = Image.fromarray(current_state['image'])
        mask_pil = Image.fromarray(current_state['combined_mask'])
        
        img_path = f"{export_dir}/image.png"
        mask_path = f"{export_dir}/mask.png"
        info_path = f"{export_dir}/info.txt"
        
        img_pil.save(img_path)
        mask_pil.save(mask_path)
        
        # Save info
        with open(info_path, 'w') as f:
            f.write(f"Export timestamp: {timestamp}\n")
            f.write(f"Image size: {img_pil.size}\n")
            f.write(f"Objects detected: {len(current_state['detections'])}\n")
            f.write(f"Mask pixels: {np.sum(current_state['combined_mask'] > 0)}\n")
        
        # Create zip file
        zip_path = f"{export_dir}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(img_path, "image.png")
            zipf.write(mask_path, "mask.png")
            zipf.write(info_path, "info.txt")
        
        return f"✅ Exported to: {zip_path}\nUse this zip file with the inpainting scripts"
        
    except Exception as e:
        return f"❌ Export failed: {str(e)}"

def create_ui():
    """Create Gradio UI for mask generation"""
    
    with gr.Blocks(
        title="Room Object Detection & Mask Generation",
        theme=gr.themes.Base(),
        css=".container { max-width: 1400px; margin: auto; }"
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🏠 Room Object Detection & Mask Generation</h1>
            <p>Step 1: Generate precise masks for object removal</p>
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
                                label="Confidence Threshold"
                            )
                            yolo_iou = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.45, step=0.05,
                                label="IoU Threshold"
                            )
                            yolo_max_det = gr.Slider(
                                minimum=10, maximum=1000, value=300, step=10,
                                label="Max Detections"
                            )
                            yolo_agnostic = gr.Checkbox(
                                label="Class-Agnostic NMS",
                                value=False
                            )
                        
                        detect_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        detection_output = gr.Image(label="Detection Results")
                        detection_info = gr.Textbox(label="Summary", lines=1)
                        detection_list = gr.Textbox(label="Detected Objects", lines=10)
            
            with gr.Tab("✂️ Mask Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        selected_objects = gr.Textbox(
                            label="Objects to Remove",
                            placeholder="Enter indices (e.g., 0,2,5) or 'all'",
                            info="Use object indices from detection results"
                        )
                        
                        with gr.Row():
                            select_all_btn = gr.Button("Select All", size="sm")
                            clear_btn = gr.Button("Clear", size="sm")
                        
                        with gr.Group():
                            gr.Markdown("### SAM Parameters")
                            sam_model = gr.Dropdown(
                                choices=["vit_b", "vit_l", "vit_h"],
                                value="vit_h",
                                label="Model Type"
                            )
                            sam_multimask = gr.Checkbox(
                                label="Multi-mask Output",
                                value=True
                            )
                            sam_dilate_kernel = gr.Slider(
                                minimum=0, maximum=50, value=10, step=5,
                                label="Mask Dilation (pixels)"
                            )
                            sam_dilate_iter = gr.Slider(
                                minimum=1, maximum=5, value=1, step=1,
                                label="Dilation Iterations"
                            )
                        
                        segment_btn = gr.Button("✂️ Generate Mask", variant="primary", size="lg")
                        export_btn = gr.Button("📦 Export for Inpainting", variant="secondary", size="lg")
                    
                    with gr.Column(scale=2):
                        mask_output = gr.Image(label="Generated Mask")
                        mask_info = gr.Textbox(label="Mask Info", lines=1)
                        export_status = gr.Textbox(label="Export Status", lines=3)
        
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
        
        def select_all():
            if current_state['detections']:
                return "all"
            return ""
        
        def clear_selection():
            return ""
        
        def on_segment(indices, model, multimask, dilate_kernel, dilate_iter):
            config = SAMConfig(
                model_type=model,
                multimask_output=multimask,
                mask_dilate_kernel=dilate_kernel,
                mask_dilate_iterations=dilate_iter
            )
            return process_segmentation(indices, config)
        
        detect_btn.click(
            on_detect,
            inputs=[input_image, yolo_model, yolo_conf, yolo_iou, yolo_max_det, yolo_agnostic],
            outputs=[detection_output, detection_info, detection_list]
        )
        
        select_all_btn.click(select_all, outputs=[selected_objects])
        clear_btn.click(clear_selection, outputs=[selected_objects])
        
        segment_btn.click(
            on_segment,
            inputs=[selected_objects, sam_model, sam_multimask, sam_dilate_kernel, sam_dilate_iter],
            outputs=[mask_output, mask_info]
        )
        
        export_btn.click(export_for_inpainting, outputs=[export_status])
    
    return demo

def check_models():
    """Check required models"""
    models_needed = [
        "yolov8x.pt", "sam_vit_h_4b8939.pth", "sam_vit_b_01ec64.pth"
    ]
    
    print("\n📦 Model Status:")
    missing = []
    for model_file in models_needed:
        path = f"models/{model_file}"
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✅ {model_file} ({size:.1f} MB)")
        else:
            print(f"  ❌ {model_file}")
            missing.append(model_file)
    
    if missing:
        print(f"\n⚠️ Missing {len(missing)} model files!")
        print("Run: python setup.py")
    
    return len(missing) == 0

if __name__ == "__main__":
    print("🚀 Starting Room Object Detection & Mask Generation")
    print(f"📍 Device: {device}")
    
    check_models()
    
    demo = create_ui()
    demo.launch(share=False, inbrowser=True)