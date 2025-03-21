#!/usr/bin/env python3
"""
Modified training pipeline script that uses gsplat for 3D Gaussian Splatting on Apple Silicon/Intel Macs.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D Gaussian model on a point cloud using gsplat")
    
    parser.add_argument("--point_cloud", type=str, required=True, 
                        help="Path to the point cloud PLY file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the trained model and results")
    parser.add_argument("--iterations", type=int, default=30000,
                        help="Number of training iterations")
    parser.add_argument("--kernel_size", type=float, default=0.1,
                        help="Initial size of Gaussian kernels")
    parser.add_argument("--cameras_json", type=str, required=True,
                        help="Path to cameras.json file with camera parameters")
    
    return parser.parse_args()

def prepare_training_folder(point_cloud_path, cameras_json, output_dir):
    """
    Prepare the training folder structure
    """
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    # Create scene folder structure
    scene_dir = os.path.join(train_dir, "scene")
    os.makedirs(scene_dir, exist_ok=True)
    
    # Copy point cloud file to scene directory
    shutil.copy(point_cloud_path, os.path.join(scene_dir, "point_cloud.ply"))
    
    # Copy cameras.json file to scene directory
    shutil.copy(cameras_json, os.path.join(scene_dir, "cameras.json"))
    
    return train_dir

def train_3d_gaussians(train_dir, output_dir, iterations, kernel_size):
    """
    Train 3D Gaussians using gsplat
    """
    # Set up command to run the gsplat training script
    cmd = [
        "python",
        "gsplat_train.py",
        f"--source_path={train_dir}",
        f"--model_path={output_dir}",
        f"--iterations={iterations}",
        f"--kernel_size={kernel_size}",
    ]
    
    # Execute the command
    print("Starting 3D Gaussian training with gsplat...")
    print(f"Command: {' '.join(cmd)}")
    os.system(" ".join(cmd))
    
    return os.path.join(output_dir, f"point_cloud/iteration_{iterations}/point_cloud.ply")

def prepare_web_viewer(trained_model_path, output_dir):
    """
    Prepare web viewer with the trained model
    """
    # Create web output directory
    web_dir = os.path.join(output_dir, "web")
    scenes_dir = os.path.join(web_dir, "scenes", "ct_scan")
    os.makedirs(scenes_dir, exist_ok=True)
    
    # Copy trained model to web viewer directory
    if os.path.exists(trained_model_path):
        print(f"Copying trained model from {trained_model_path} to {scenes_dir}")
        shutil.copy(trained_model_path, os.path.join(scenes_dir, "point_cloud.ply"))
    else:
        print(f"Warning: Trained model not found at {trained_model_path}")
        # Try to find it in another location or use the original point cloud
        print("Using original point cloud instead")
        
    # Copy cameras.json
    cameras_json = os.path.join(os.path.dirname(os.path.dirname(trained_model_path)), "cameras.json")
    if os.path.exists(cameras_json):
        shutil.copy(cameras_json, os.path.join(scenes_dir, "cameras.json"))
    else:
        print(f"Warning: cameras.json not found at {cameras_json}")
    
    # Look for the viewer files in web-splat directory
    web_splat_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web-splat")
    if os.path.exists(web_splat_dir):
        # Copy necessary files from web-splat
        print(f"Copying web viewer files from {web_splat_dir}")
        
        # Check for demo.html or index.html
        if os.path.exists(os.path.join(web_splat_dir, "public", "demo.html")):
            shutil.copy(os.path.join(web_splat_dir, "public", "demo.html"), os.path.join(web_dir, "viewer.html"))
        elif os.path.exists(os.path.join(web_splat_dir, "public", "index.html")):
            shutil.copy(os.path.join(web_splat_dir, "public", "index.html"), os.path.join(web_dir, "viewer.html"))
        
        # Create static directory and copy static files
        static_dir = os.path.join(web_dir, "static")
        os.makedirs(static_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(web_splat_dir, "public", "static")):
            for item in os.listdir(os.path.join(web_splat_dir, "public", "static")):
                src = os.path.join(web_splat_dir, "public", "static", item)
                dst = os.path.join(static_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy(src, dst)
    else:
        print(f"Warning: web-splat directory not found at {web_splat_dir}")
        print("Using existing simple_viewer.html instead")
    
    # Create redirect index.html
    with open(os.path.join(web_dir, "index.html"), "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>CT Scan Viewer</title>
    <meta http-equiv="refresh" content="0; url='simple_viewer.html?file=./scenes/ct_scan/point_cloud.ply&scene=./scenes/ct_scan/cameras.json'">
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin-top: 50px;
            background-color: #f0f0f0;
            color: #333;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #00369f;
        }
        p {
            line-height: 1.6;
        }
        a {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            background-color: #00369f;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        a:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>CT Scan 3D Viewer</h1>
        <p>Redirecting to the interactive 3D viewer for your CT scan data...</p>
        <p>If you are not redirected automatically, please click the button below.</p>
        <a href="simple_viewer.html?file=./scenes/ct_scan/point_cloud.ply&scene=./scenes/ct_scan/cameras.json">Open 3D Viewer</a>
    </div>
</body>
</html>""")
    
    return web_dir

def main():
    """
    Main function to run the pipeline
    """
    args = parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare training folder
    train_dir = prepare_training_folder(args.point_cloud, args.cameras_json, args.output_dir)
    
    # Train 3D Gaussians
    trained_model_path = train_3d_gaussians(
        train_dir, 
        os.path.join(args.output_dir, "model"),
        args.iterations,
        args.kernel_size
    )
    
    # Prepare web viewer
    web_dir = prepare_web_viewer(trained_model_path, args.output_dir)
    
    print("\nTraining pipeline completed!")
    print(f"Output directory: {args.output_dir}")
    print(f"Web viewer: {web_dir}")
    print("\nYou can view the results by running a web server in the output directory:")
    print(f"cd {args.output_dir} && python -m http.server 8000")
    print("Then open http://localhost:8000/web/ in your browser.")

if __name__ == "__main__":
    main() 