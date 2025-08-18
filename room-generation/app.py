#!/usr/bin/env python3
"""
Video to 3D Model Reconstruction App
Unified application with Gradio UI and COLMAP integration
"""

import os
import sys
import cv2
import time
import json
import shutil
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
import urllib.request
import zipfile

import gradio as gr

# Optional visualization
try:
    import trimesh
    import plotly.graph_objects as go
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

class Video3DReconstructor:
    """Complete video to 3D reconstruction pipeline"""
    
    PRESETS = {
        "fast": {"fps": 1.0, "blur_threshold": 30, "image_size": 1024, "features": 2048},
        "balanced": {"fps": 2.0, "blur_threshold": 50, "image_size": 1920, "features": 4096},
        "quality": {"fps": 3.0, "blur_threshold": 70, "image_size": 2560, "features": 8192}
    }
    
    def __init__(self):
        self.pipeline = None
        self.log = []
        
    def setup_colmap(self):
        """Ensure COLMAP is available"""
        colmap_exe = Path("tools/COLMAP/COLMAP-3.9.1-windows-cuda/bin/colmap.exe")
        
        if colmap_exe.exists():
            os.environ["PATH"] = str(colmap_exe.parent) + os.pathsep + os.environ.get("PATH", "")
            lib_path = colmap_exe.parent.parent / "lib"
            if lib_path.exists():
                os.environ["PATH"] = str(lib_path) + os.pathsep + os.environ.get("PATH", "")
            return True
        
        self.log_msg("COLMAP not found. Downloading...")
        try:
            os.makedirs("tools", exist_ok=True)
            url = "https://github.com/colmap/colmap/releases/download/3.9.1/COLMAP-3.9.1-windows-cuda.zip"
            zip_path = "colmap_temp.zip"
            
            def download_progress(block_num, block_size, total_size):
                percent = min(block_num * block_size * 100 / total_size, 100)
                sys.stdout.write(f'\rDownloading: {percent:.1f}%')
                sys.stdout.flush()
            
            urllib.request.urlretrieve(url, zip_path, download_progress)
            self.log_msg("Extracting COLMAP...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("tools/COLMAP/")
            os.remove(zip_path)
            
            os.environ["PATH"] = str(colmap_exe.parent) + os.pathsep + os.environ.get("PATH", "")
            lib_path = colmap_exe.parent.parent / "lib"
            if lib_path.exists():
                os.environ["PATH"] = str(lib_path) + os.pathsep + os.environ.get("PATH", "")
            
            return True
        except Exception as e:
            self.log_msg(f"Failed to setup COLMAP: {e}")
            return False
    
    def log_msg(self, msg):
        """Add timestamped message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {msg}")
        if len(self.log) > 20:
            self.log.pop(0)
    
    def extract_frames(self, video_path, output_dir, preset):
        """Extract quality frames from video"""
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps / preset["fps"]))
        
        self.log_msg(f"Video: {fps:.1f} FPS, {total_frames} frames, sampling every {frame_interval} frames")
        
        frame_count = 0
        saved_count = 0
        prev_gray = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                is_different = True
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    if np.mean(diff) < 5:
                        is_different = False
                
                if blur_score >= preset["blur_threshold"] and is_different:
                    h, w = frame.shape[:2]
                    max_size = preset["image_size"]
                    if max(h, w) > max_size:
                        scale = max_size / max(h, w)
                        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                    
                    cv2.imwrite(str(frames_dir / f"frame_{saved_count:05d}.jpg"), frame, 
                               [cv2.IMWRITE_JPEG_QUALITY, 90])
                    saved_count += 1
                    prev_gray = gray
            
            frame_count += 1
        
        cap.release()
        self.log_msg(f"Extracted {saved_count} quality frames")
        return saved_count, frames_dir
    
    def run_colmap(self, frames_dir, workspace_dir, preset):
        """Run COLMAP 3D reconstruction"""
        database = workspace_dir / "database.db"
        sparse_dir = workspace_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        
        try:
            # Feature extraction
            self.log_msg("Extracting features...")
            cmd = [
                "colmap", "feature_extractor",
                "--database_path", str(database),
                "--image_path", str(frames_dir),
                "--SiftExtraction.use_gpu", "0",
                "--SiftExtraction.max_image_size", str(preset["image_size"]),
                "--SiftExtraction.max_num_features", str(preset["features"])
            ]
            
            if sys.platform == "win32":
                subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
            else:
                subprocess.run(cmd, capture_output=True, text=True)
            
            # Feature matching
            self.log_msg("Matching features...")
            cmd = [
                "colmap", "exhaustive_matcher",
                "--database_path", str(database),
                "--SiftMatching.use_gpu", "0"
            ]
            
            if sys.platform == "win32":
                subprocess.run(" ".join(cmd), shell=True, capture_output=True)
            else:
                subprocess.run(cmd, capture_output=True)
            
            # Sparse reconstruction
            self.log_msg("Creating 3D model...")
            cmd = [
                "colmap", "mapper",
                "--database_path", str(database),
                "--image_path", str(frames_dir),
                "--output_path", str(sparse_dir)
            ]
            
            if sys.platform == "win32":
                subprocess.run(" ".join(cmd), shell=True, capture_output=True)
            else:
                subprocess.run(cmd, capture_output=True)
            
            self.log_msg("Reconstruction complete!")
            return True
            
        except Exception as e:
            self.log_msg(f"COLMAP error: {e}")
            return sparse_dir.exists() and any(sparse_dir.iterdir())
    
    def export_results(self, workspace_dir, model_dir):
        """Export reconstruction results"""
        output_files = []
        sparse_dir = workspace_dir / "sparse"
        
        if sparse_dir.exists():
            sparse_model = sparse_dir / "0"
            if not sparse_model.exists():
                models = list(sparse_dir.iterdir())
                if models:
                    sparse_model = models[0]
            
            if sparse_model.exists():
                # Export as text format
                txt_dir = model_dir / "sparse_txt"
                txt_dir.mkdir(exist_ok=True)
                
                cmd = [
                    "colmap", "model_converter",
                    "--input_path", str(sparse_model),
                    "--output_path", str(txt_dir),
                    "--output_type", "TXT"
                ]
                
                try:
                    if sys.platform == "win32":
                        subprocess.run(" ".join(cmd), shell=True, capture_output=True)
                    else:
                        subprocess.run(cmd, capture_output=True)
                except:
                    pass
                
                # Export as PLY point cloud
                ply_path = model_dir / "point_cloud.ply"
                cmd = [
                    "colmap", "model_converter",
                    "--input_path", str(sparse_model),
                    "--output_path", str(ply_path),
                    "--output_type", "PLY"
                ]
                
                try:
                    if sys.platform == "win32":
                        subprocess.run(" ".join(cmd), shell=True, capture_output=True)
                    else:
                        subprocess.run(cmd, capture_output=True)
                    if ply_path.exists():
                        output_files.append(str(ply_path))
                except:
                    pass
                
                # Create zip of binary model
                zip_path = model_dir / "colmap_model.zip"
                shutil.make_archive(str(zip_path.with_suffix('')), 'zip', sparse_model)
                output_files.append(str(zip_path))
                
                # Create simple text point cloud for easy viewing
                self.export_simple_pointcloud(sparse_model, model_dir)
                xyz_path = model_dir / "points.xyz"
                if xyz_path.exists():
                    output_files.append(str(xyz_path))
        
        return output_files
    
    def export_simple_pointcloud(self, sparse_model, model_dir):
        """Export simple XYZ point cloud file"""
        try:
            points_file = sparse_model / "points3D.txt"
            if not points_file.exists():
                points_file = sparse_model / "points3D.bin"
                if not points_file.exists():
                    return
            
            xyz_path = model_dir / "points.xyz"
            with open(xyz_path, 'w') as out:
                if points_file.suffix == '.txt':
                    with open(points_file, 'r') as f:
                        for line in f:
                            if line.startswith('#'):
                                continue
                            parts = line.strip().split()
                            if len(parts) >= 7:
                                x, y, z = parts[1:4]
                                r, g, b = parts[4:7]
                                out.write(f"{x} {y} {z} {r} {g} {b}\n")
        except:
            pass
    
    def create_preview(self, frames_dir):
        """Create frames preview grid"""
        try:
            frame_files = sorted(list(frames_dir.glob("*.jpg")))[:9]
            if not frame_files:
                return None
            
            images = []
            for f in frame_files:
                img = cv2.imread(str(f))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (200, 150))
                images.append(img)
            
            while len(images) < 9:
                images.append(np.zeros((150, 200, 3), dtype=np.uint8))
            
            rows = []
            for i in range(0, 9, 3):
                row = np.hstack(images[i:i+3])
                rows.append(row)
            
            return np.vstack(rows)
        except:
            return None
    
    def create_3d_preview(self, workspace_dir):
        """Create 3D point cloud visualization"""
        if not VIZ_AVAILABLE:
            return None
            
        try:
            points_file = workspace_dir / "sparse" / "0" / "points3D.txt"
            if not points_file.exists():
                return None
            
            points = []
            with open(points_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            if not points:
                return None
            
            points = np.array(points)
            if len(points) > 5000:
                indices = np.random.choice(len(points), 5000, replace=False)
                points = points[indices]
            
            fig = go.Figure(data=[go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(size=2, color=points[:, 2], colorscale='Viridis', opacity=0.8)
            )])
            
            fig.update_layout(
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                title="3D Point Cloud", height=600
            )
            
            return fig
        except:
            return None
    
    def process(self, video_file, quality, progress=gr.Progress()):
        """Main processing pipeline"""
        self.log = []
        
        if not video_file:
            return "Please upload a video", None, None, None, ""
        
        try:
            # Setup
            video_path = Path(video_file.name if hasattr(video_file, 'name') else video_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"output_{video_path.stem}_{timestamp}")
            workspace_dir = output_dir / "workspace"
            model_dir = output_dir / "model"
            
            for d in [workspace_dir, model_dir]:
                d.mkdir(parents=True, exist_ok=True)
            
            preset = self.PRESETS.get(quality.lower(), self.PRESETS["fast"])
            
            # Setup COLMAP
            progress(0.1, desc="Setting up...")
            self.log_msg("Setting up COLMAP...")
            if not self.setup_colmap():
                return "Failed to setup COLMAP", None, None, None, "\n".join(self.log)
            
            # Extract frames
            progress(0.2, desc="Extracting frames...")
            self.log_msg("Extracting frames from video...")
            num_frames, frames_dir = self.extract_frames(video_path, output_dir, preset)
            
            if num_frames < 10:
                return "Too few frames extracted", None, None, None, "\n".join(self.log)
            
            # Create preview
            frames_preview = self.create_preview(frames_dir)
            
            # Run reconstruction
            progress(0.5, desc="Reconstructing 3D model...")
            self.log_msg("Starting 3D reconstruction...")
            success = self.run_colmap(frames_dir, workspace_dir, preset)
            
            if not success:
                return "Reconstruction failed", frames_preview, None, None, "\n".join(self.log)
            
            # Export and create previews
            progress(0.9, desc="Finalizing...")
            files = self.export_results(workspace_dir, model_dir)
            preview_3d = self.create_3d_preview(workspace_dir)
            
            # Summary
            summary = f"""
            ## ✅ Reconstruction Complete!
            
            **Stats:**
            - Frames extracted: {num_frames}
            - Quality preset: {quality}
            - Output folder: {output_dir}
            
            **📁 Generated Files:**
            - `point_cloud.ply` - 3D point cloud (open in MeshLab/CloudCompare)
            - `points.xyz` - Simple text format (open in any 3D viewer)
            - `colmap_model.zip` - Full COLMAP model
            
            **🔍 How to View Your 3D Model:**
            
            1. **MeshLab** (Free, recommended):
               - Download: https://www.meshlab.net/#download
               - Open the .ply file directly
            
            2. **CloudCompare** (Free, professional):
               - Download: https://cloudcompare.org/
               - Open .ply or .xyz files
            
            3. **Online Viewer** (No install):
               - Go to: https://3dviewer.net/
               - Drag & drop the .ply file
            
            4. **COLMAP GUI** (Advanced):
               - Open colmap_model.zip contents in COLMAP GUI
            """
            
            progress(1.0, desc="Done!")
            self.log_msg("Process complete!")
            
            return summary, frames_preview, preview_3d, files, "\n".join(self.log)
            
        except Exception as e:
            self.log_msg(f"Error: {str(e)}")
            return f"Error: {str(e)}", None, None, None, "\n".join(self.log)

def create_ui():
    """Create Gradio interface"""
    reconstructor = Video3DReconstructor()
    
    with gr.Blocks(title="Video to 3D Model", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📹 Video to 3D Model Reconstruction")
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.File(
                    label="Upload Video (MP4/MOV/AVI)",
                    file_types=[".mp4", ".mov", ".avi", ".MOV", ".MP4", ".AVI"]
                )
                quality = gr.Radio(
                    ["Fast", "Balanced", "Quality"],
                    value="Fast",
                    label="Quality",
                    info="Higher quality = longer processing"
                )
                process_btn = gr.Button("🚀 Start Reconstruction", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                output = gr.Markdown("Ready to process")
                log = gr.Textbox(label="Log", lines=8, max_lines=20)
        
        with gr.Row():
            frames_preview = gr.Image(label="Extracted Frames", type="numpy")
            if VIZ_AVAILABLE:
                preview_3d = gr.Plot(label="3D Preview")
            else:
                preview_3d = gr.Markdown("Install `trimesh plotly` for 3D preview")
        
        download = gr.File(label="Download 3D Model", file_count="multiple")
        
        process_btn.click(
            fn=reconstructor.process,
            inputs=[video_input, quality],
            outputs=[output, frames_preview, preview_3d, download, log],
            show_progress=True
        )
        
        gr.Markdown("""
        ### 📝 Tips for Best Results:
        - Record 30-60 second videos with slow, steady camera movement
        - Capture all angles of the room/object
        - Ensure good lighting and avoid motion blur
        - Use "Fast" mode for quick tests, "Quality" for final models
        """)
    
    return app

if __name__ == "__main__":
    app = create_ui()
    print("\n" + "="*50)
    print("Video to 3D Model Reconstruction")
    print("="*50)
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)