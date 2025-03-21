import os
import argparse
import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Set up the environment for the training process."""
    # Check if the cinematic-gaussians repository is available
    if not os.path.exists('../cinematic-gaussians'):
        print("Error: cinematic-gaussians repository not found.")
        print("Please clone the repository first:")
        print("git clone https://github.com/KeKsBoTer/cinematic-gaussians.git --recursive")
        sys.exit(1)
    
    # We'll use the current Python environment instead of creating a new conda environment
    print("Using current Python environment...")
    
    # Verify required packages are installed
    try:
        import numpy
        import torch
        import scipy
        import tqdm
        import PIL
        print("Required packages are installed.")
    except ImportError as e:
        print(f"Error: Missing package: {e}")
        print("Please install required packages with:")
        print("pip install numpy torch scipy tqdm pillow")
        sys.exit(1)
    
    print("Environment setup complete.")

def train_model(scene_folder, model_output_folder, iterations=30000):
    """Train the 3D Gaussian Splatting model."""
    print(f"Training model with data from {scene_folder}...")
    
    # Create the model output folder
    os.makedirs(model_output_folder, exist_ok=True)
    
    # Build the training command - use Python directly
    train_cmd = [
        sys.executable,  # Use the same Python interpreter
        '../cinematic-gaussians/train.py',
        '-s', scene_folder,
        '-m', model_output_folder,
        '--eval',
        '--test_iterations', '7000', '15000', str(iterations),
        '--densify_grad_threshold', '0.00005',
        '--save_iterations', str(iterations)
    ]
    
    # Run the training command
    try:
        subprocess.run(train_cmd, check=True)
        print(f"Training complete. Model saved to {model_output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        print("Training failed, but continuing with the pipeline.")

def compress_model(model_folder, compressed_output_folder, iteration=30000):
    """Compress the trained model for efficient web viewing."""
    print(f"Compressing model from {model_folder}...")
    
    # Create the compressed output folder
    os.makedirs(compressed_output_folder, exist_ok=True)
    
    # Build the compression command - use Python directly
    compress_cmd = [
        sys.executable,  # Use the same Python interpreter
        '../cinematic-gaussians/compress.py',
        '-m', model_folder,
        '--eval',
        '--output_vq', compressed_output_folder,
        '--load_iteration', str(iteration)
    ]
    
    # Run the compression command
    try:
        subprocess.run(compress_cmd, check=True)
        print(f"Compression complete. Compressed model saved to {compressed_output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error during compression: {e}")
        print("Compression failed, but continuing with the pipeline.")

def prepare_web_viewer(compressed_model_folder, web_output_folder):
    """Prepare the web viewer files for the compressed model."""
    print(f"Preparing web viewer for compressed model from {compressed_model_folder}...")
    
    # Create the web output folder
    os.makedirs(web_output_folder, exist_ok=True)
    
    # Copy the compressed model files to the web output folder
    model_file = os.path.join(compressed_model_folder, "point_cloud/iteration_30000/point_cloud.npz")
    camera_file = os.path.join(compressed_model_folder, "cameras.json")
    
    if not os.path.exists(model_file):
        print(f"Error: Could not find compressed model file at {model_file}")
        # Use the original point cloud instead
        original_ply = os.path.join(compressed_model_folder, "..", "scene", "point_cloud.ply")
        if os.path.exists(original_ply):
            print(f"Using original point cloud file at {original_ply} instead")
            # Create necessary directories
            os.makedirs(os.path.join(compressed_model_folder, "point_cloud/iteration_30000"), exist_ok=True)
            # Copy the file
            subprocess.run(['cp', original_ply, model_file.replace(".npz", ".ply")])
            model_file = model_file.replace(".npz", ".ply")
        else:
            print("Could not find original point cloud either. Web viewer may not work properly.")
            return
    
    if not os.path.exists(camera_file):
        print(f"Error: Could not find camera file at {camera_file}")
        # Look for camera file in scene folder
        alt_camera_file = os.path.join(compressed_model_folder, "..", "scene", "cameras.json")
        if os.path.exists(alt_camera_file):
            print(f"Using camera file from scene folder: {alt_camera_file}")
            subprocess.run(['cp', alt_camera_file, camera_file])
        else:
            print("Could not find camera file. Web viewer may not work properly.")
            return
    
    # Create scenes directory in web output folder
    scenes_dir = os.path.join(web_output_folder, "scenes/ct_scan")
    os.makedirs(scenes_dir, exist_ok=True)
    
    # Copy files to the scenes directory
    subprocess.run(['cp', model_file, os.path.join(scenes_dir, os.path.basename(model_file))])
    subprocess.run(['cp', camera_file, os.path.join(scenes_dir, "cameras.json")])
    
    # Check if web-splat repository is available
    if not os.path.exists('../web-splat'):
        print("Error: web-splat repository not found.")
        print("Please clone the repository first:")
        print("git clone https://github.com/KeKsBoTer/web-splat.git")
        return
    
    # Copy the necessary web-splat files to the web output folder
    # Use the demo.html file as the viewer
    subprocess.run(['cp', '../web-splat/public/demo.html', os.path.join(web_output_folder, "viewer.html")])
    
    # Create static directory in web output folder for JavaScript and CSS
    static_dir = os.path.join(web_output_folder, "static")
    os.makedirs(static_dir, exist_ok=True)
    
    # Copy the static files
    subprocess.run(['cp', '-r', '../web-splat/public/static', web_output_folder])
    
    # Create an index.html file that redirects to the viewer with our scene
    model_filename = os.path.basename(model_file)
    with open(os.path.join(web_output_folder, "index.html"), 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0; url='viewer.html?file=./scenes/ct_scan/{model_filename}&scene=./scenes/ct_scan/cameras.json'">
    <title>CT Cinematic Viewer</title>
</head>
<body>
    <p>Redirecting to viewer...</p>
</body>
</html>
""")
    
    print(f"Web viewer prepared in {web_output_folder}")
    print(f"You can view the model by opening {os.path.join(web_output_folder, 'index.html')} in a browser")

def serve_web_viewer(web_output_folder, port=8000):
    """Start a simple HTTP server to serve the web viewer."""
    print(f"Starting HTTP server to serve web viewer from {web_output_folder} on port {port}...")
    
    # Change to the web output folder
    os.chdir(web_output_folder)
    
    # Start the HTTP server
    subprocess.run([sys.executable, '-m', 'http.server', str(port)])

def main():
    parser = argparse.ArgumentParser(description="Train and compress a 3D Gaussian model from a point cloud")
    parser.add_argument("--scene_folder", required=True, help="Folder containing point cloud and camera data")
    parser.add_argument("--output_dir", required=True, help="Base output directory")
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training iterations")
    parser.add_argument("--serve", action="store_true", help="Start a web server after preparation")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web server")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and compression")
    
    args = parser.parse_args()
    
    # Create output directories
    model_output_folder = os.path.join(args.output_dir, "model")
    compressed_output_folder = os.path.join(args.output_dir, "compressed")
    web_output_folder = os.path.join(args.output_dir, "web")
    
    # Setup environment
    setup_environment()
    
    if not args.skip_training:
        # Train model
        train_model(args.scene_folder, model_output_folder, args.iterations)
        
        # Compress model
        compress_model(model_output_folder, compressed_output_folder, args.iterations)
    
    # Prepare web viewer
    prepare_web_viewer(compressed_output_folder, web_output_folder)
    
    # Serve web viewer if requested
    if args.serve:
        serve_web_viewer(web_output_folder, args.port)

if __name__ == "__main__":
    main() 