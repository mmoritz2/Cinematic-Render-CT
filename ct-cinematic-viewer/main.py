#!/usr/bin/env python3
import os
import argparse
import sys
from pathlib import Path

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dicom_processor import process_dicom_to_pointcloud
from train_compress import setup_environment, train_model, compress_model, prepare_web_viewer, serve_web_viewer
# Import the gsplat training module
from train_gsplat import prepare_training_folder, train_3d_gaussians, prepare_web_viewer as gsplat_prepare_web_viewer

def main():
    parser = argparse.ArgumentParser(description="DICOM to Cinematic CT Viewer Pipeline")
    
    # Input/output options
    parser.add_argument("--dicom_dir", required=True, help="Directory containing DICOM files")
    parser.add_argument("--output_dir", required=True, help="Base output directory")
    
    # DICOM processing options
    parser.add_argument("--threshold", type=int, default=250, help="HU threshold for segmentation")
    parser.add_argument("--num_points", type=int, default=100000, help="Number of points to sample")
    parser.add_argument("--num_cameras", type=int, default=50, help="Number of camera views")
    
    # Training options
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training iterations")
    parser.add_argument("--kernel_size", type=float, default=0.1, help="Initial kernel size for Gaussians")
    
    # Workflow options
    parser.add_argument("--skip_dicom", action="store_true", help="Skip DICOM processing")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and compression")
    parser.add_argument("--serve", action="store_true", help="Start a web server after preparation")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web server")
    
    # Added rendering engine option
    parser.add_argument("--use_gsplat", action="store_true", help="Use gsplat for training (Apple Silicon/Intel compatible)")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    scene_folder = os.path.join(args.output_dir, "scene")
    model_output_folder = os.path.join(args.output_dir, "model")
    compressed_output_folder = os.path.join(args.output_dir, "compressed")
    web_output_folder = os.path.join(args.output_dir, "web")
    
    print("=== CT to Cinematic Rendering Pipeline ===")
    
    # Step 1: Process DICOM to Point Cloud
    if not args.skip_dicom:
        print("\n=== Step 1: Processing DICOM files ===")
        point_cloud_file, camera_file = process_dicom_to_pointcloud(
            args.dicom_dir, 
            scene_folder, 
            args.threshold, 
            args.num_points, 
            args.num_cameras
        )
        print(f"Point cloud created at: {point_cloud_file}")
        print(f"Camera file created at: {camera_file}")
    else:
        print("\n=== Step 1: Skipping DICOM processing ===")
        print(f"Assuming point cloud and camera data already exists in: {scene_folder}")
        # Make sure the files exist
        point_cloud_file = os.path.join(scene_folder, "point_cloud.ply")
        camera_file = os.path.join(scene_folder, "cameras.json")
        if not os.path.exists(point_cloud_file) or not os.path.exists(camera_file):
            print(f"Error: Could not find point_cloud.ply or cameras.json in {scene_folder}")
            print("Please run without --skip_dicom or ensure the files exist")
            sys.exit(1)
    
    # Step 2: Setup environment
    print("\n=== Step 2: Setting up environment ===")
    if not args.use_gsplat:
        setup_environment()
    else:
        print("Using gsplat for training (compatible with Apple Silicon/Intel)")
    
    # Step 3 & 4: Train and compress 3D Gaussian model
    if not args.skip_training:
        if not args.use_gsplat:
            # Standard CUDA-based training and compression
            print("\n=== Step 3: Training 3D Gaussian model (CUDA) ===")
            train_model(scene_folder, model_output_folder, args.iterations)
            
            print("\n=== Step 4: Compressing model ===")
            compress_model(model_output_folder, compressed_output_folder, args.iterations)
            
            # Step 5: Prepare web viewer
            print("\n=== Step 5: Preparing web viewer ===")
            prepare_web_viewer(compressed_output_folder, web_output_folder)
        else:
            # gsplat-based training (Apple Silicon/Intel compatible)
            print("\n=== Step 3: Training 3D Gaussian model (gsplat) ===")
            # Prepare training folder
            train_dir = prepare_training_folder(point_cloud_file, camera_file, args.output_dir)
            
            # Train model
            trained_model_path = train_3d_gaussians(
                train_dir, 
                os.path.join(args.output_dir, "model"),
                args.iterations,
                args.kernel_size
            )
            
            # Step 4: Skip compression since gsplat produces optimized output
            print("\n=== Step 4: Skipping separate compression step (not needed with gsplat) ===")
            
            # Step 5: Prepare web viewer with gsplat output
            print("\n=== Step 5: Preparing web viewer ===")
            gsplat_prepare_web_viewer(trained_model_path, args.output_dir)
    else:
        print("\n=== Steps 3 & 4: Skipping training and compression ===")
        if not args.use_gsplat:
            print(f"Assuming compressed model already exists in: {compressed_output_folder}")
        else:
            print(f"Assuming trained model already exists in: {model_output_folder}")
        
        # Step 5: Prepare web viewer
        print("\n=== Step 5: Preparing web viewer ===")
        if not args.use_gsplat:
            prepare_web_viewer(compressed_output_folder, web_output_folder)
        else:
            # Look for the trained model
            potential_paths = [
                os.path.join(model_output_folder, f"point_cloud/iteration_{args.iterations}/point_cloud.ply"),
                os.path.join(model_output_folder, "point_cloud/iteration_final/point_cloud.ply")
            ]
            
            trained_model_path = None
            for path in potential_paths:
                if os.path.exists(path):
                    trained_model_path = path
                    break
            
            if trained_model_path:
                gsplat_prepare_web_viewer(trained_model_path, args.output_dir)
            else:
                print(f"Warning: Could not find trained model. Using original point cloud.")
                gsplat_prepare_web_viewer(point_cloud_file, args.output_dir)
    
    # Step 6: Serve web viewer if requested
    if args.serve:
        print("\n=== Step 6: Starting web server ===")
        print(f"Access the viewer at: http://localhost:{args.port}")
        serve_web_viewer(web_output_folder, args.port)
    else:
        print("\n=== Pipeline complete ===")
        print(f"To view the rendered CT data, open: {os.path.join(web_output_folder, 'index.html')}")
        print(f"Or run: python -m http.server {args.port} --directory {web_output_folder}")
        print(f"And access: http://localhost:{args.port}")

if __name__ == "__main__":
    main()