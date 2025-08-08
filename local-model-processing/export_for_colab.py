"""
Export images and masks for Google Colab processing
Integrated with room_removal_ultimate.py for seamless workflow
"""
import os
import shutil
import cv2
import numpy as np
from datetime import datetime
import zipfile

def export_for_colab(image, mask, export_dir="colab_export", create_zip=True):
    """
    Export image and mask in format ready for Colab processing
    
    Args:
        image: numpy array (BGR format from OpenCV)
        mask: numpy array (single channel mask)
        export_dir: base directory for exports
        create_zip: whether to create a zip file for easy upload
    """
    # Create export directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(export_dir, timestamp)
    os.makedirs(export_path, exist_ok=True)
    
    # Save image
    image_path = os.path.join(export_path, "image.png")
    cv2.imwrite(image_path, image)
    
    # Save mask
    mask_path = os.path.join(export_path, "mask.png")
    cv2.imwrite(mask_path, mask)
    
    # Create info file
    info_path = os.path.join(export_path, "info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Export for Colab Processing\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Image shape: {image.shape}\n")
        f.write(f"Mask shape: {mask.shape}\n")
        f.write(f"\nQuick Instructions:\n")
        f.write(f"1. Upload the ZIP file to Google Colab\n")
        f.write(f"2. Use lama_refinement_optimized.ipynb\n")
        f.write(f"3. Run with GPU for best quality refinement\n")
        f.write(f"\nNote: Only LaMa model needed on Colab (no YOLO/SAM required)\n")
    
    # Create zip if requested
    zip_path = None
    if create_zip:
        zip_path = f"{export_path}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(image_path, "image.png")
            zipf.write(mask_path, "mask.png")
            zipf.write(info_path, "info.txt")
    
    print(f"[OK] Exported to: {export_path}")
    print(f"   - {image_path}")
    print(f"   - {mask_path}")
    if zip_path:
        print(f"   - {zip_path} (ready for upload)")
    
    print(f"\n[INFO] Next steps:")
    print(f"1. Upload {os.path.basename(zip_path)} to Google Colab")
    print(f"2. Open lama_refinement_optimized.ipynb")
    print(f"3. Run with GPU runtime for best quality")
    print(f"\n[TIP] The optimized notebook only needs LaMa model (no YOLO/SAM required)")
    
    return export_path


def export_batch(image_mask_pairs, export_dir="colab_export_batch"):
    """
    Export multiple image/mask pairs for batch processing
    
    Args:
        image_mask_pairs: list of (image, mask) tuples
        export_dir: directory for batch export
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(export_dir, f"batch_{timestamp}")
    os.makedirs(os.path.join(batch_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(batch_dir, "masks"), exist_ok=True)
    
    for i, (image, mask) in enumerate(image_mask_pairs):
        cv2.imwrite(os.path.join(batch_dir, "images", f"image_{i:03d}.png"), image)
        cv2.imwrite(os.path.join(batch_dir, "masks", f"image_{i:03d}_mask.png"), mask)
    
    # Create zip
    zip_path = f"{batch_dir}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(batch_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, batch_dir)
                zipf.write(file_path, arcname)
    
    print(f"[OK] Batch exported: {len(image_mask_pairs)} pairs")
    print(f"[ZIP] Zip file: {zip_path}")
    return batch_dir


# Integration function for room_removal_ultimate.py
def add_export_button_to_ui(current_state):
    """
    Add this to your Gradio UI for easy export
    Call after mask generation
    """
    def export_current():
        if current_state.get('image') is not None and current_state.get('combined_mask') is not None:
            export_path = export_for_colab(
                current_state['image'],
                current_state['combined_mask'],
                create_zip=True
            )
            return f"Exported to: {export_path}.zip"
        return "No image/mask to export"
    
    return export_current


if __name__ == "__main__":
    print("Export utility for Colab processing")
    print("\nUsage:")
    print("1. Import in room_removal_ultimate.py")
    print("2. Call export_for_colab(image, mask) after mask generation")
    print("3. Upload exported files to Google Drive")
    print("4. Process with LaMa on Colab (GPU)")
    print("\nBenefits:")
    print("- No need to upload YOLO/SAM models to Colab")
    print("- Faster uploads (just image + mask)")
    print("- Can batch process multiple images")