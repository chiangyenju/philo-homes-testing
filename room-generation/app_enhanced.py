#!/usr/bin/env python3
"""
Enhanced Video to 3D Model Reconstruction
With GPU support, dense reconstruction, and optimized parameters
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
import platform

import gradio as gr

# Optional visualization
try:
    import trimesh
    import plotly.graph_objects as go
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

class EnhancedVideo3D:
    """Enhanced reconstruction pipeline with better point density"""
    
    PRESETS = {
        "fast": {
            "fps": 2.0, 
            "blur_threshold": 20, 
            "image_size": 1920, 
            "features": 8192,
            "matcher": "exhaustive",
            "dense": False
        },
        "balanced": {
            "fps": 3.0, 
            "blur_threshold": 30, 
            "image_size": 2560, 
            "features": 16384,
            "matcher": "exhaustive",
            "dense": True
        },
        "quality": {
            "fps": 5.0, 
            "blur_threshold": 40, 
            "image_size": 3840, 
            "features": 32768,
            "matcher": "exhaustive",
            "dense": True
        }
    }
    
    def __init__(self):
        self.pipeline = None
        self.log = []
        self.gpu_available = self.check_gpu()
        
    def check_gpu(self):
        """Check if GPU/CUDA is available"""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                self.log_msg("NVIDIA GPU detected - enabling GPU acceleration")
                return True
        except:
            pass
        self.log_msg("No GPU detected - using CPU (slower but works)")
        return False
        
    def setup_colmap(self):
        """Setup COLMAP with proper version"""
        # Try to find existing COLMAP
        colmap_exe = Path("tools/COLMAP/COLMAP-3.9.1-windows-cuda/bin/colmap.exe")
        
        if colmap_exe.exists():
            os.environ["PATH"] = str(colmap_exe.parent) + os.pathsep + os.environ.get("PATH", "")
            lib_path = colmap_exe.parent.parent / "lib"
            if lib_path.exists():
                os.environ["PATH"] = str(lib_path) + os.pathsep + os.environ.get("PATH", "")
            return True
        
        self.log_msg("Downloading COLMAP...")
        try:
            os.makedirs("tools", exist_ok=True)
            # Use CUDA version even for CPU - it includes both
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
        if len(self.log) > 30:
            self.log.pop(0)
    
    def extract_frames(self, video_path, output_dir, preset):
        """Extract high-quality frames with better coverage"""
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps / preset["fps"]))
        
        self.log_msg(f"Video: {fps:.1f} FPS, {total_frames} frames")
        self.log_msg(f"Extracting every {frame_interval} frames for better coverage")
        
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
                
                # Less strict difference check for more frames
                is_different = True
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    if np.mean(diff) < 3:  # Lower threshold
                        is_different = False
                
                if blur_score >= preset["blur_threshold"] and is_different:
                    h, w = frame.shape[:2]
                    max_size = preset["image_size"]
                    if max(h, w) > max_size:
                        scale = max_size / max(h, w)
                        frame = cv2.resize(frame, (int(w*scale), int(h*scale)), 
                                         interpolation=cv2.INTER_LANCZOS4)
                    
                    # Save with high quality
                    cv2.imwrite(str(frames_dir / f"frame_{saved_count:05d}.jpg"), frame, 
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_count += 1
                    prev_gray = gray
            
            frame_count += 1
        
        cap.release()
        self.log_msg(f"Extracted {saved_count} high-quality frames")
        return saved_count, frames_dir
    
    def run_colmap_enhanced(self, frames_dir, workspace_dir, preset):
        """Enhanced COLMAP pipeline with better parameters"""
        database = workspace_dir / "database.db"
        sparse_dir = workspace_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        
        use_gpu = "1" if self.gpu_available else "0"
        
        try:
            # 1. Feature extraction with more features
            self.log_msg("Extracting features (this may take a while)...")
            cmd = [
                "colmap", "feature_extractor",
                "--database_path", str(database),
                "--image_path", str(frames_dir),
                "--ImageReader.camera_model", "OPENCV",
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu", use_gpu,
                "--SiftExtraction.max_image_size", str(preset["image_size"]),
                "--SiftExtraction.max_num_features", str(preset["features"]),
                "--SiftExtraction.first_octave", "-1",
                "--SiftExtraction.num_octaves", "4",
                "--SiftExtraction.octave_resolution", "3",
                "--SiftExtraction.peak_threshold", "0.004",
                "--SiftExtraction.edge_threshold", "10",
                "--SiftExtraction.estimate_affine_shape", "1",
                "--SiftExtraction.domain_size_pooling", "1"
            ]
            
            if sys.platform == "win32":
                result = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
            
            if "error" in result.stderr.lower():
                self.log_msg("Feature extraction warning - continuing...")
            
            # 2. Feature matching with exhaustive or sequential
            self.log_msg("Matching features across frames...")
            
            if preset["matcher"] == "exhaustive":
                cmd = [
                    "colmap", "exhaustive_matcher",
                    "--database_path", str(database),
                    "--SiftMatching.use_gpu", use_gpu,
                    "--SiftMatching.max_ratio", "0.8",
                    "--SiftMatching.max_distance", "0.7",
                    "--SiftMatching.cross_check", "1",
                    "--SiftMatching.max_num_matches", "32768"
                ]
            else:
                cmd = [
                    "colmap", "sequential_matcher",
                    "--database_path", str(database),
                    "--SiftMatching.use_gpu", use_gpu,
                    "--SequentialMatching.overlap", "10",
                    "--SequentialMatching.loop_detection", "1"
                ]
            
            if sys.platform == "win32":
                subprocess.run(" ".join(cmd), shell=True, capture_output=True)
            else:
                subprocess.run(cmd, capture_output=True)
            
            # 3. Sparse reconstruction with better parameters
            self.log_msg("Creating sparse 3D model...")
            cmd = [
                "colmap", "mapper",
                "--database_path", str(database),
                "--image_path", str(frames_dir),
                "--output_path", str(sparse_dir),
                "--Mapper.ba_refine_focal_length", "1",
                "--Mapper.ba_refine_principal_point", "1",
                "--Mapper.ba_refine_extra_params", "1",
                "--Mapper.min_num_matches", "15",
                "--Mapper.init_min_num_inliers", "100",
                "--Mapper.abs_pose_min_num_inliers", "30",
                "--Mapper.abs_pose_min_inlier_ratio", "0.25",
                "--Mapper.ba_local_max_num_iterations", "25",
                "--Mapper.ba_global_max_num_iterations", "50",
                "--Mapper.ba_global_images_ratio", "1.0",
                "--Mapper.ba_global_points_ratio", "1.0",
                "--Mapper.ba_global_max_refinements", "5",
                "--Mapper.extract_colors", "1"
            ]
            
            if sys.platform == "win32":
                subprocess.run(" ".join(cmd), shell=True, capture_output=True)
            else:
                subprocess.run(cmd, capture_output=True)
            
            # 4. Dense reconstruction if requested
            if preset["dense"] and sparse_dir.exists():
                self.log_msg("Creating dense reconstruction...")
                self.run_dense_reconstruction(frames_dir, sparse_dir, workspace_dir, use_gpu)
            
            self.log_msg("Reconstruction complete!")
            return True
            
        except Exception as e:
            self.log_msg(f"COLMAP error: {e}")
            return sparse_dir.exists() and any(sparse_dir.iterdir())
    
    def run_dense_reconstruction(self, frames_dir, sparse_dir, workspace_dir, use_gpu):
        """Run dense reconstruction for more points"""
        try:
            dense_dir = workspace_dir / "dense"
            dense_dir.mkdir(exist_ok=True)
            
            # Find sparse model
            sparse_model = sparse_dir / "0"
            if not sparse_model.exists():
                models = list(sparse_dir.iterdir())
                if models:
                    sparse_model = models[0]
            
            if not sparse_model.exists():
                return
            
            # Undistort images
            self.log_msg("Undistorting images...")
            cmd = [
                "colmap", "image_undistorter",
                "--image_path", str(frames_dir),
                "--input_path", str(sparse_model),
                "--output_path", str(dense_dir),
                "--output_type", "COLMAP"
            ]
            
            if sys.platform == "win32":
                subprocess.run(" ".join(cmd), shell=True, capture_output=True)
            else:
                subprocess.run(cmd, capture_output=True)
            
            # Patch match stereo
            self.log_msg("Running stereo matching...")
            cmd = [
                "colmap", "patch_match_stereo",
                "--workspace_path", str(dense_dir),
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.geom_consistency", "true",
                "--PatchMatchStereo.gpu_index", "0" if use_gpu == "1" else "-1",
                "--PatchMatchStereo.window_radius", "5",
                "--PatchMatchStereo.num_samples", "15",
                "--PatchMatchStereo.num_iterations", "5"
            ]
            
            if sys.platform == "win32":
                subprocess.run(" ".join(cmd), shell=True, capture_output=True)
            else:
                subprocess.run(cmd, capture_output=True)
            
            # Stereo fusion
            self.log_msg("Fusing depth maps...")
            output_ply = dense_dir / "fused.ply"
            cmd = [
                "colmap", "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--workspace_format", "COLMAP",
                "--input_type", "geometric",
                "--output_path", str(output_ply)
            ]
            
            if sys.platform == "win32":
                subprocess.run(" ".join(cmd), shell=True, capture_output=True)
            else:
                subprocess.run(cmd, capture_output=True)
            
            if output_ply.exists():
                shutil.copy(output_ply, workspace_dir.parent / "model" / "dense_cloud.ply")
                self.log_msg("Dense reconstruction created!")
                
        except Exception as e:
            self.log_msg(f"Dense reconstruction error: {e}")
    
    def export_results(self, workspace_dir, model_dir):
        """Export all reconstruction results"""
        output_files = []
        
        # Check for dense cloud first
        dense_ply = model_dir / "dense_cloud.ply"
        if dense_ply.exists():
            output_files.append(str(dense_ply))
            self.log_msg(f"Dense cloud exported: {dense_ply.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Export sparse model
        sparse_dir = workspace_dir / "sparse"
        if sparse_dir.exists():
            sparse_model = sparse_dir / "0"
            if not sparse_model.exists():
                models = list(sparse_dir.iterdir())
                if models:
                    sparse_model = models[0]
            
            if sparse_model.exists():
                # Export as PLY
                ply_path = model_dir / "sparse_cloud.ply"
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
                
                # Export as text
                txt_dir = model_dir / "colmap_text"
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
                
                # Create model zip
                zip_path = model_dir / "colmap_model.zip"
                shutil.make_archive(str(zip_path.with_suffix('')), 'zip', sparse_model)
                output_files.append(str(zip_path))
                
                # Export camera parameters
                self.export_camera_info(sparse_model, model_dir)
        
        return output_files
    
    def export_camera_info(self, sparse_model, model_dir):
        """Export camera parameters for debugging"""
        try:
            cameras_file = sparse_model / "cameras.txt"
            if cameras_file.exists():
                shutil.copy(cameras_file, model_dir / "cameras.txt")
            
            # Create reconstruction info
            info = {
                "timestamp": datetime.now().isoformat(),
                "gpu_used": self.gpu_available,
                "log": self.log[-20:]  # Last 20 log entries
            }
            
            with open(model_dir / "reconstruction_info.json", "w") as f:
                json.dump(info, f, indent=2)
        except:
            pass
    
    def create_preview(self, frames_dir):
        """Create frames preview grid"""
        try:
            frame_files = sorted(list(frames_dir.glob("*.jpg")))[:12]
            if not frame_files:
                return None
            
            images = []
            for f in frame_files:
                img = cv2.imread(str(f))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (160, 120))
                images.append(img)
            
            while len(images) < 12:
                images.append(np.zeros((120, 160, 3), dtype=np.uint8))
            
            rows = []
            for i in range(0, 12, 4):
                row = np.hstack(images[i:i+4])
                rows.append(row)
            
            return np.vstack(rows)
        except:
            return None
    
    def process(self, video_file, quality, progress=gr.Progress()):
        """Main processing pipeline"""
        self.log = []
        
        if not video_file:
            return "Please upload a video", None, None, ""
        
        try:
            # Setup
            video_path = Path(video_file.name if hasattr(video_file, 'name') else video_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"output_{video_path.stem}_{timestamp}")
            workspace_dir = output_dir / "workspace"
            model_dir = output_dir / "model"
            
            for d in [workspace_dir, model_dir]:
                d.mkdir(parents=True, exist_ok=True)
            
            preset = self.PRESETS.get(quality.lower(), self.PRESETS["balanced"])
            
            # Setup COLMAP
            progress(0.1, desc="Setting up...")
            self.log_msg(f"Initializing with {quality} preset...")
            if not self.setup_colmap():
                return "Failed to setup COLMAP", None, None, "\n".join(self.log)
            
            # Extract frames
            progress(0.2, desc="Extracting frames...")
            self.log_msg("Extracting frames from video...")
            num_frames, frames_dir = self.extract_frames(video_path, output_dir, preset)
            
            if num_frames < 20:
                self.log_msg(f"Warning: Only {num_frames} frames extracted. Try:")
                self.log_msg("- Recording a longer video (60+ seconds)")
                self.log_msg("- Moving camera more slowly")
                self.log_msg("- Ensuring better lighting")
            
            # Create preview
            frames_preview = self.create_preview(frames_dir)
            
            # Run enhanced reconstruction
            progress(0.4, desc="Running 3D reconstruction...")
            self.log_msg("Starting enhanced 3D reconstruction...")
            success = self.run_colmap_enhanced(frames_dir, workspace_dir, preset)
            
            if not success:
                self.log_msg("Reconstruction failed - check video quality")
                return "Reconstruction failed", frames_preview, None, "\n".join(self.log)
            
            # Export
            progress(0.9, desc="Exporting models...")
            files = self.export_results(workspace_dir, model_dir)
            
            # Summary
            dense_exists = (model_dir / "dense_cloud.ply").exists()
            point_count = "Dense" if dense_exists else "Sparse"
            
            summary = f"""
            ## ✅ Enhanced Reconstruction Complete!
            
            **Stats:**
            - Frames processed: {num_frames}
            - Quality: {quality}
            - GPU acceleration: {'Yes' if self.gpu_available else 'No'}
            - Point cloud type: {point_count}
            - Output: {output_dir}
            
            **📁 Files Generated:**
            - `dense_cloud.ply` - Dense point cloud (if available)
            - `sparse_cloud.ply` - Sparse point cloud
            - `colmap_model.zip` - Full COLMAP model
            
            **🔍 View in:**
            - **MeshLab**: https://www.meshlab.net/#download
            - **CloudCompare**: https://cloudcompare.org/
            - **Online**: https://3dviewer.net/
            
            **💡 Tips for Better Results:**
            - Use Quality mode for maximum points
            - Record 60+ second videos
            - Move camera slowly with overlap
            - Ensure all surfaces are well-lit
            - Avoid reflective/textureless surfaces
            """
            
            progress(1.0, desc="Done!")
            self.log_msg("Process complete!")
            
            return summary, frames_preview, files, "\n".join(self.log)
            
        except Exception as e:
            self.log_msg(f"Error: {str(e)}")
            return f"Error: {str(e)}", None, None, "\n".join(self.log)

def create_ui():
    """Create enhanced Gradio interface"""
    reconstructor = EnhancedVideo3D()
    
    with gr.Blocks(title="Enhanced 3D Reconstruction", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🚀 Enhanced Video to 3D Reconstruction
        ### Now with GPU support, dense reconstruction, and optimized parameters
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.File(
                    label="Upload Video (MP4/MOV/AVI)",
                    file_types=[".mp4", ".mov", ".avi", ".MOV", ".MP4", ".AVI"]
                )
                quality = gr.Radio(
                    ["Fast", "Balanced", "Quality"],
                    value="Balanced",
                    label="Quality Mode",
                    info="Quality mode creates dense point clouds"
                )
                process_btn = gr.Button("🎯 Start Enhanced Reconstruction", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 📋 Requirements for Best Results:
                - **Video length**: 60+ seconds
                - **Movement**: Slow, overlapping views
                - **Lighting**: Bright, even illumination
                - **Coverage**: Capture all surfaces
                - **GPU**: NVIDIA GPU recommended (but not required)
                """)
            
            with gr.Column(scale=2):
                output = gr.Markdown("Ready to process")
                frames_preview = gr.Image(label="Frame Samples", type="numpy")
                download = gr.File(label="Download 3D Models", file_count="multiple")
                log = gr.Textbox(label="Processing Log", lines=10, max_lines=30)
        
        process_btn.click(
            fn=reconstructor.process,
            inputs=[video_input, quality],
            outputs=[output, frames_preview, download, log],
            show_progress=True
        )
        
        gr.Markdown("""
        ### 🛠️ Improvements in this version:
        - **More features extracted** (up to 32K per image)
        - **Better matching algorithms**
        - **Dense reconstruction** in Balanced/Quality modes
        - **GPU acceleration** when available
        - **Optimized COLMAP parameters** for room-scale scenes
        - **Higher quality frame extraction**
        """)
    
    return app

if __name__ == "__main__":
    app = create_ui()
    print("\n" + "="*60)
    print("Enhanced 3D Reconstruction - Maximum Point Density")
    print("="*60)
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)