#!/usr/bin/env python3
"""
Dependency installer for SAM2 + LaMa notebook
Handles all dependency conflicts automatically
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

def main():
    print("🔧 Installing dependencies...")
    
    # Step 1: Upgrade pip
    print("1/6 Upgrading pip...")
    install_package("--upgrade pip")
    
    # Step 2: Install core dependencies with specific versions
    print("2/6 Installing core packages...")
    install_package("--upgrade pillow>=10.1")
    install_package("--upgrade numpy>=2.0.0")
    
    # Step 3: Install image processing libraries
    print("3/6 Installing image processing libraries...")
    install_package("opencv-python")
    install_package("matplotlib")
    install_package("scipy")
    install_package("scikit-image")
    
    # Step 4: Install UI libraries
    print("4/6 Installing UI libraries...")
    install_package("ipywidgets")
    install_package("ipycanvas")
    
    # Step 5: Install PyTorch and SAM2
    print("5/6 Installing PyTorch and SAM2...")
    install_package("torch")
    install_package("torchvision")
    install_package("git+https://github.com/facebookresearch/sam2.git")
    
    # Step 6: Install LaMa
    print("6/6 Installing LaMa...")
    install_package("simple-lama-inpainting")
    
    print("✅ All dependencies installed successfully!")
    print("You can now run the notebook without errors.")

if __name__ == "__main__":
    main()