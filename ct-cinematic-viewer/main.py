#!/usr/bin/env python3
import os
import argparse
import sys
from pathlib import Path

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dicom_processor import process_dicom_to_pointcloud
from train_compress import setup_environment, train_model, compress_model, prepare_web_viewer, serve_web_viewer
from cloud_gpu import prepare_cloud_job, upload_data, download_results

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
    
    # Workflow options
    parser.add_argument("--skip_dicom", action="store_true", help="Skip DICOM processing")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and compression")
    parser.add_argument("--serve", action="store_true", help="Start a web server after preparation")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web server")
    
    # GPU options
    parser.add_argument("--use_cloud", action="store_true", help="Use cloud GPU instead of local GPU")
    parser.add_argument("--cloud_provider", type=str, default="aws", choices=["aws", "gcp", "azure"], 
                        help="Cloud provider to use")
    parser.add_argument("--cloud_instance", type=str, default="g4dn.xlarge", 
                        help="Cloud instance type (AWS: g4dn.xlarge, GCP: n1-standard-4-nvidia-tesla-t4)")
    parser.add_argument("--cloud_region", type=str, default="us-west-2", 
                        help="Cloud region to use")
    
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
    if not args.use_cloud:
        setup_environment()
    else:
        print(f"Setting up cloud environment on {args.cloud_provider}...")
    
    # Step 3 & 4: Train and compress 3D Gaussian model
    if not args.skip_training:
        if not args.use_cloud:
            # Local GPU training
            print("\n=== Step 3: Training 3D Gaussian model (local GPU) ===")
            train_model(scene_folder, model_output_folder, args.iterations)
            
            print("\n=== Step 4: Compressing model ===")
            compress_model(model_output_folder, compressed_output_folder, args.iterations)
        else:
            # Cloud GPU training
            print(f"\n=== Step 3: Training 3D Gaussian model (cloud GPU - {args.cloud_provider}) ===")
            # Create and set up cloud instance
            instance_id = prepare_cloud_job(
                args.cloud_provider, 
                args.cloud_instance,
                args.cloud_region
            )
            
            # Upload data to cloud
            print(f"Uploading data to {args.cloud_provider}...")
            upload_data(
                args.cloud_provider,
                instance_id,
                scene_folder,
                args.iterations
            )
            
            # Run training on cloud
            print(f"Running training on {args.cloud_provider} (this may take several hours)...")
            # This would be implemented in the cloud_gpu.py module
            # The function would monitor the job and wait for completion
            
            # Download results from cloud
            print(f"Downloading results from {args.cloud_provider}...")
            download_results(
                args.cloud_provider,
                instance_id,
                model_output_folder,
                compressed_output_folder
            )
            
            print(f"Cloud resources on {args.cloud_provider} have been released.")
    else:
        print("\n=== Steps 3 & 4: Skipping training and compression ===")
        print(f"Assuming compressed model already exists in: {compressed_output_folder}")
    
    # Step 5: Prepare web viewer
    print("\n=== Step 5: Preparing web viewer ===")
    prepare_web_viewer(compressed_output_folder, web_output_folder)
    
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