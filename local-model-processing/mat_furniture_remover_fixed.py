#!/usr/bin/env python3
"""
MAT Furniture Remover - FIXED VERSION
Properly implements MAT's mask convention: 0=inpaint, 1=keep
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import zipfile
import tempfile
import shutil

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not available. Use command-line mode.")

# Add MAT repository to path
MAT_PATH = "models/mat/MAT"
if os.path.exists(MAT_PATH):
    sys.path.insert(0, MAT_PATH)

try:
    import legacy
    from networks.mat import Generator
    print("✅ MAT modules loaded")
except ImportError as e:
    print(f"❌ MAT modules not found: {e}")
    sys.exit(1)


class MATFurnitureRemover:
    def __init__(self, model_path="models/mat/pretrained/Places_512_FullData.pkl"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load MAT model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MAT model not found at {self.model_path}")
        
        print(f"Loading MAT model...")
        print(f"Using device: {self.device}")
        
        # Override torch.load for CPU
        original_load = torch.load
        if self.device.type == 'cpu':
            torch.load = lambda *args, **kwargs: original_load(*args, map_location='cpu', **{k:v for k,v in kwargs.items() if k != 'map_location'})
        
        try:
            with open(self.model_path, 'rb') as f:
                network_dict = legacy.load_network_pkl(f)
                self.model = network_dict['G_ema']
                if self.device.type != 'cpu':
                    self.model = self.model.to(self.device)
        finally:
            if self.device.type == 'cpu':
                torch.load = original_load
        
        print("✅ MAT model loaded!")
    
    def process_image_and_mask(self, image, mask, dilate_mask=15):
        """Process image and mask with proper MAT format"""
        
        # Ensure RGB and L mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Process mask - dilate to include edges/shadows
        mask_np = np.array(mask)
        
        if dilate_mask > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (dilate_mask*2+1, dilate_mask*2+1))
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        
        # CRITICAL: Invert mask for MAT
        # MAT expects: 0 = inpaint (remove furniture), 1 = keep
        # Our mask: 255 = furniture to remove, 0 = keep
        mask_np = 255 - mask_np  # Invert the mask
        
        mask = Image.fromarray(mask_np)
        
        return image, mask
    
    def pad_to_multiple(self, image, mask, multiple=512):
        """Resize to exactly 512x512 for MAT"""
        # MAT works best with square 512x512 or 1024x1024 images
        # Let's resize to 512x512 to avoid dimension mismatch
        
        orig_w, orig_h = image.size
        
        # For MAT, use fixed 512x512 resolution
        target_size = 512
        
        # Resize maintaining aspect ratio, then pad to square
        aspect = orig_w / orig_h
        
        if aspect > 1:  # Wider than tall
            new_w = target_size
            new_h = int(target_size / aspect)
        else:  # Taller than wide
            new_h = target_size
            new_w = int(target_size * aspect)
        
        # Resize first
        image = image.resize((new_w, new_h), Image.LANCZOS)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        
        # Create square canvas
        padded_img = Image.new('RGB', (target_size, target_size), (128, 128, 128))
        padded_mask = Image.new('L', (target_size, target_size), 255)  # 255 = keep
        
        # Center the image
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        
        padded_img.paste(image, (x_offset, y_offset))
        padded_mask.paste(mask, (x_offset, y_offset))
        
        padding_info = (orig_w, orig_h, x_offset, y_offset, new_w, new_h)
        
        return padded_img, padded_mask, padding_info
    
    def inpaint(self, image, mask):
        """Run MAT inpainting with multiple passes and refinement"""
        
        # Store original size and image
        original_size = image.size
        original_image = image.copy()
        
        # Process mask (dilate and invert)
        image, mask = self.process_image_and_mask(image, mask, dilate_mask=25)
        
        # Store processed mask for blending
        processed_mask = mask.copy()
        
        # Pad to multiple of 512
        image, mask, padding_info = self.pad_to_multiple(image, mask, 512)
        
        # Convert to tensors
        image_np = np.array(image).transpose(2, 0, 1).astype(np.float32)
        image_tensor = torch.from_numpy(image_np).float().to(self.device) / 127.5 - 1
        image_tensor = image_tensor.unsqueeze(0)
        
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).float().to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        
        print("Running enhanced MAT inpainting...")
        
        # Run multiple inference passes with different seeds for better quality
        results = []
        num_passes = 5  # More passes for better selection
        
        # Use varied truncation and noise modes for diversity
        truncation_values = [0.7, 0.85, 1.0, 0.6, 0.75]
        noise_modes = ['const', 'random', 'random', 'const', 'random']
        
        for i in range(num_passes):
            # Set different random seed for each pass
            torch.manual_seed(240 + i * 100)
            np.random.seed(240 + i * 100)
            
            # Generate different latent code for variety
            z = torch.randn(1, self.model.z_dim).to(self.device)
            
            # No class labels for unconditional model
            label = torch.zeros([1, self.model.c_dim]).to(self.device) if self.model.c_dim > 0 else None
            
            # Use predetermined truncation values
            truncation = truncation_values[i]
            noise_mode = noise_modes[i]
            
            # Run model
            with torch.no_grad():
                output = self.model(image_tensor, mask_tensor, z, label, 
                                  truncation_psi=truncation, noise_mode=noise_mode)
            
            # Convert output
            output_np = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            output_np = output_np[0].cpu().numpy()
            
            # Score based on smoothness and color consistency
            score = self.score_inpainting(output_np, image_np, mask_np)
            results.append((output_np, score, truncation, noise_mode))
            print(f"  Pass {i+1}: score={score:.2f}, truncation={truncation:.2f}, noise={noise_mode}")
        
        # Select best result
        results.sort(key=lambda x: x[1])
        best_result, best_score, best_trunc, best_noise = results[0]
        print(f"  Selected: score={best_score:.2f}, truncation={best_trunc:.2f}, noise={best_noise}")
        
        result = Image.fromarray(best_result, 'RGB')
        
        # Crop back if padded
        if padding_info:
            orig_w, orig_h, x_offset, y_offset, new_w, new_h = padding_info
            result = result.crop((x_offset, y_offset, x_offset + new_w, y_offset + new_h))
        
        # Resize to original
        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)
        
        # Apply post-processing refinements
        result = self.refine_result(original_image, result, processed_mask)
        
        return result
    
    def score_inpainting(self, output, original, mask):
        """Score the inpainting quality based on multiple factors"""
        # 1. Gradient smoothness in masked regions
        output_gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(output_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(output_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Focus on masked regions and boundary
        mask_binary = mask > 0.5
        if len(mask_binary.shape) > 2:
            mask_binary = mask_binary[:,:,0]
        
        # Expand mask slightly to include boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_expanded = cv2.dilate(mask_binary.astype(np.uint8), kernel, iterations=1)
        
        # Calculate gradient score (lower is better)
        masked_gradient = gradient_mag * mask_expanded
        gradient_score = np.mean(masked_gradient[mask_expanded > 0])
        
        # 2. Color consistency with surrounding areas
        # Get border region colors
        border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_dilated = cv2.dilate(mask_binary.astype(np.uint8), border_kernel, iterations=1)
        border_mask = mask_dilated - mask_binary.astype(np.uint8)
        
        if np.sum(border_mask) > 100:
            # Original image in CHW format, convert to HWC for border sampling
            original_hwc = np.transpose(original, (1, 2, 0))
            border_colors = original_hwc[border_mask > 0]
            inpainted_colors = output[mask_binary > 0]
            
            if len(border_colors) > 0 and len(inpainted_colors) > 0:
                # Calculate color difference
                border_mean = np.mean(border_colors, axis=0)
                border_std = np.std(border_colors, axis=0)
                inpainted_mean = np.mean(inpainted_colors, axis=0)
                inpainted_std = np.std(inpainted_colors, axis=0)
                
                # Color consistency score (lower is better)
                color_diff = np.mean(np.abs(border_mean - inpainted_mean))
                std_diff = np.mean(np.abs(border_std - inpainted_std))
                color_score = color_diff + std_diff * 0.5
            else:
                color_score = 0
        else:
            color_score = 0
        
        # 3. Check for artifacts (very bright or dark spots)
        inpainted_region = output[mask_binary > 0]
        if len(inpainted_region) > 0:
            # Detect extreme values
            very_bright = np.sum(np.all(inpainted_region > 250, axis=1))
            very_dark = np.sum(np.all(inpainted_region < 5, axis=1))
            artifact_score = (very_bright + very_dark) / len(inpainted_region) * 100
        else:
            artifact_score = 0
        
        # Combine scores (weighted)
        total_score = gradient_score + color_score * 0.3 + artifact_score * 2
        return total_score
    
    def refine_result(self, original, inpainted, mask):
        """Apply advanced post-processing refinements for better quality"""
        original_np = np.array(original)
        inpainted_np = np.array(inpainted)
        mask_np = np.array(mask)
        
        # 1. Advanced edge feathering with multi-scale blending
        mask_float = mask_np.astype(np.float32) / 255.0
        
        # Invert mask back for blending (remember we inverted it for MAT)
        mask_float = 1.0 - mask_float
        
        # Multi-scale soft masks for progressive blending
        soft_mask1 = cv2.GaussianBlur(mask_float, (11, 11), 0)
        soft_mask2 = cv2.GaussianBlur(mask_float, (21, 21), 0)
        soft_mask3 = cv2.GaussianBlur(mask_float, (31, 31), 0)
        
        # Combine multi-scale masks for smoother transition
        soft_mask = (soft_mask1 * 0.3 + soft_mask2 * 0.5 + soft_mask3 * 0.2)
        soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)
        
        # Initial blend with soft edges
        result = original_np * (1 - soft_mask_3ch) + inpainted_np * soft_mask_3ch
        
        # 2. Advanced color correction using histogram matching
        if np.sum(mask_float) > 100:  # Only if mask is significant
            # Get colors from border region
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            dilated = cv2.dilate(mask_float, kernel, iterations=1)
            border_mask = dilated - mask_float
            
            if np.sum(border_mask) > 100:
                # Sample colors from border
                border_pixels = original_np[border_mask > 0.3]
                inpainted_pixels = result[mask_float > 0.5]
                
                if len(border_pixels) > 0 and len(inpainted_pixels) > 0:
                    # Calculate statistics for each channel
                    result = result.astype(np.float32)
                    
                    for c in range(3):
                        border_channel = border_pixels[:, c]
                        inpainted_channel = result[mask_float > 0.5, c]
                        
                        # Match mean and std
                        border_mean = np.mean(border_channel)
                        border_std = np.std(border_channel)
                        inpaint_mean = np.mean(inpainted_channel)
                        inpaint_std = np.std(inpainted_channel) + 1e-6
                        
                        # Normalize and rescale
                        result[mask_float > 0.5, c] = (result[mask_float > 0.5, c] - inpaint_mean) / inpaint_std
                        result[mask_float > 0.5, c] = result[mask_float > 0.5, c] * border_std + border_mean
                    
                    # Apply adjustment gradually based on distance from border
                    dist_transform = cv2.distanceTransform((mask_float > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
                    dist_norm = np.clip(dist_transform / (np.max(dist_transform) + 1e-6), 0, 1)
                    dist_weight = 1.0 - np.exp(-dist_norm * 2)  # Stronger adjustment further from border
                    dist_weight_3ch = np.stack([dist_weight] * 3, axis=-1)
                    
                    # Blend original and corrected based on distance
                    corrected = result.copy()
                    result = inpainted_np * (1 - dist_weight_3ch) + corrected * dist_weight_3ch
                    result = np.clip(result, 0, 255)
        
        # 3. Adaptive sharpening based on local detail
        result_uint8 = result.astype(np.uint8)
        
        # Use unsharp mask for better control
        gaussian = cv2.GaussianBlur(result_uint8, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(result_uint8, 1.5, gaussian, -0.5, 0)
        
        # Detect edges to apply sharpening selectively
        edges = cv2.Canny(result_uint8, 50, 150)
        edge_mask = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (5, 5), 0)
        edge_mask_3ch = np.stack([edge_mask] * 3, axis=-1)
        
        # Apply sharpening more to edges, less to smooth areas
        sharpening_strength = soft_mask_3ch * edge_mask_3ch * 0.7
        result = result_uint8 * (1 - sharpening_strength) + unsharp_mask * sharpening_strength
        
        # 4. Final Poisson blending for seamless integration
        try:
            # Create mask for Poisson blending
            poisson_mask = (mask_float > 0.1).astype(np.uint8) * 255
            
            # Find center of masked region for Poisson blending
            moments = cv2.moments(poisson_mask)
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Apply Poisson blending
                result = cv2.seamlessClone(
                    result.astype(np.uint8),
                    original_np.astype(np.uint8),
                    poisson_mask,
                    (cx, cy),
                    cv2.NORMAL_CLONE
                )
        except:
            # If Poisson blending fails, keep the previous result
            pass
        
        return Image.fromarray(result.astype(np.uint8))
    
    def process_zip(self, zip_file):
        """Process zip file from room_removal_ultimate.py"""
        
        if zip_file is None:
            return None, None, "Please upload a zip file"
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Handle Gradio file object
            # In Gradio 3.50.2, file comes as a tempfile._TemporaryFileWrapper
            if hasattr(zip_file, 'name'):
                zip_path = zip_file.name
            elif isinstance(zip_file, str):
                zip_path = zip_file
            else:
                # For Gradio file objects, save to temp location
                import tempfile as tf
                with tf.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                    tmp.write(zip_file.read() if hasattr(zip_file, 'read') else zip_file)
                    zip_path = tmp.name
            
            # Extract zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find image and mask
            image_path = None
            mask_path = None
            
            for file in os.listdir(temp_dir):
                if 'image' in file.lower() and file.endswith(('.png', '.jpg')):
                    image_path = os.path.join(temp_dir, file)
                elif 'mask' in file.lower() and file.endswith(('.png', '.jpg')):
                    mask_path = os.path.join(temp_dir, file)
            
            if not image_path or not mask_path:
                return None, None, "Could not find image and mask in zip"
            
            # Load images
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Inpaint
            result = self.inpaint(image, mask)
            
            # Create comparison
            comparison = self.create_comparison(image, mask, result)
            
            return result, comparison, "✅ Furniture removed successfully!"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, f"Error: {str(e)}"
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def create_comparison(self, original, mask, result):
        """Create side-by-side comparison"""
        width, height = original.size
        
        # Create canvas
        comparison = Image.new('RGB', (width * 3, height))
        
        # Original
        comparison.paste(original, (0, 0))
        
        # Mask visualization (show what will be removed in red)
        mask_viz = original.copy()
        mask_np = np.array(mask)
        overlay = np.array(mask_viz)
        # Show furniture to be removed in red
        overlay[mask_np > 128] = [255, 0, 0]
        mask_viz = Image.fromarray(overlay)
        comparison.paste(mask_viz, (width, 0))
        
        # Result
        comparison.paste(result, (width * 2, 0))
        
        return comparison


def create_gradio_ui():
    """Create Gradio interface"""
    remover = MATFurnitureRemover()
    
    with gr.Blocks(title="MAT Furniture Remover", theme=gr.themes.Base()) as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🪑➜✨ MAT Furniture Remover</h1>
            <p>Remove furniture and fill with environment using MAT</p>
            <p style="color: #666;">Upload ZIP from room_removal_ultimate.py</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                zip_input = gr.File(
                    label="Upload ZIP file",
                    file_types=[".zip"],
                    type="file"  # Changed from "filepath" to "file" for Gradio 3.50.2
                )
                
                process_btn = gr.Button("🎨 Remove Furniture", variant="primary", size="lg")
                
                status = gr.Textbox(label="Status", lines=2)
                
                gr.HTML("""
                <div style="margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 5px;">
                    <h4>📋 Instructions:</h4>
                    <ol style="text-align: left;">
                        <li>Export from room_removal_ultimate.py</li>
                        <li>Upload the ZIP file here</li>
                        <li>Click Remove Furniture</li>
                        <li>Save the result!</li>
                    </ol>
                </div>
                """)
            
            with gr.Column(scale=2):
                result_image = gr.Image(label="Clean Result", type="pil")
                comparison_image = gr.Image(label="Before | Masked | After", type="pil")
        
        # Process function
        def process(zip_file):
            if zip_file is None:
                return None, None, "Please upload a ZIP file"
            try:
                # Debug: Check what type of object we received
                print(f"Received file type: {type(zip_file)}")
                print(f"File attributes: {dir(zip_file)}")
                
                # In Gradio 3.50.2, the file is a tempfile._TemporaryFileWrapper
                # We need to get its name property
                if hasattr(zip_file, 'name'):
                    # Use the temporary file path directly
                    file_path = zip_file.name
                    print(f"Using file path: {file_path}")
                    
                    # Verify it's a zip file
                    import zipfile
                    if not zipfile.is_zipfile(file_path):
                        # Try reading as bytes and saving
                        with open(file_path, 'rb') as f:
                            data = f.read()
                        
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                            tmp.write(data)
                            tmp_path = tmp.name
                        
                        result, comparison, status = remover.process_zip(tmp_path)
                        os.unlink(tmp_path)
                        return result, comparison, status
                    else:
                        return remover.process_zip(file_path)
                else:
                    return remover.process_zip(zip_file)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, f"Error: {str(e)}"
        
        process_btn.click(
            process,
            inputs=[zip_input],
            outputs=[result_image, comparison_image, status]
        )
    
    return demo


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MAT Furniture Remover")
    parser.add_argument("zip_file", nargs="?", help="ZIP file from room_removal_ultimate.py")
    parser.add_argument("--output", default="furniture_removed.png", help="Output file")
    
    args = parser.parse_args()
    
    if args.zip_file:
        # Command-line mode
        print("Processing:", args.zip_file)
        remover = MATFurnitureRemover()
        
        # Process zip
        result, comparison, status = remover.process_zip(args.zip_file)
        
        if result:
            result.save(args.output)
            comparison.save(args.output.replace('.png', '_comparison.png'))
            print(f"✅ Saved: {args.output}")
        else:
            print(f"❌ {status}")
    
    elif GRADIO_AVAILABLE:
        # Gradio UI mode
        print("Starting Gradio UI...")
        demo = create_gradio_ui()
        print("Launching Gradio UI...")
        demo.launch(share=False, inbrowser=True)
    
    else:
        print("Usage:")
        print("  python mat_furniture_remover_fixed.py <zip_file> [--output result.png]")
        print("\nInstall Gradio for web UI:")
        print("  pip install gradio")


if __name__ == "__main__":
    print("="*60)
    print("MAT FURNITURE REMOVER (FIXED)")
    print("="*60)
    main()