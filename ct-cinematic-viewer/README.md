# CT Cinematic Viewer

A pipeline for converting CT DICOM data into cinematic 3D renderings using 3D Gaussian Splatting.

## Features

- Process DICOM files into point clouds
- Train 3D Gaussian Splatting models for high-quality rendering
- Compress models for efficient web viewing
- Serve interactive web viewer for exploring 3D CT data
- Support for both local GPU and cloud GPU processing

## Requirements

- Python 3.8+
- CUDA-capable GPU (for local processing)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository and its submodules:
   ```bash
   git clone https://github.com/mmoritz2/Cinematic-Render-CT.git --recursive
   cd Cinematic-Render-CT
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Clone required repositories:
   ```bash
   git clone https://github.com/KeKsBoTer/cinematic-gaussians.git
   git clone https://github.com/KeKsBoTer/web-splat.git
   ```

## Usage

### Basic Workflow

```bash
python main.py --dicom_dir "/path/to/dicom/files" --output_dir "/path/to/output" --serve
```

This will:
1. Process DICOM files into a point cloud
2. Train a 3D Gaussian Splatting model
3. Compress the model for web viewing
4. Start a web server to view the result

### Using Cloud GPU

For processing without a local GPU, you can use cloud-based GPUs:

```bash
python main.py --dicom_dir "/path/to/dicom/files" --output_dir "/path/to/output" --use_cloud --cloud_provider aws --cloud_instance g4dn.xlarge --serve
```

This will:
1. Process DICOM files locally
2. Create a cloud instance for GPU processing
3. Upload the point cloud data to the cloud
4. Train and compress the model on the cloud GPU
5. Download the results
6. Set up a local web viewer

### Command Line Options

#### Basic Options
- `--dicom_dir`: Directory containing DICOM files (required)
- `--output_dir`: Base output directory (required)
- `--threshold`: HU threshold for segmentation (default: 250)
- `--num_points`: Number of points to sample (default: 100000)
- `--num_cameras`: Number of camera views (default: 50)
- `--iterations`: Number of training iterations (default: 30000)

#### Workflow Options
- `--skip_dicom`: Skip DICOM processing
- `--skip_training`: Skip training and compression
- `--serve`: Start a web server after preparation
- `--port`: Port for the web server (default: 8000)

#### Cloud GPU Options
- `--use_cloud`: Use cloud GPU instead of local GPU
- `--cloud_provider`: Cloud provider to use (aws, gcp, azure) (default: aws)
- `--cloud_instance`: Cloud instance type (default: g4dn.xlarge)
- `--cloud_region`: Cloud region to use (default: us-west-2)

## Cloud GPU Support

The application supports using cloud-based GPUs from multiple providers:

### AWS
- Uses EC2 GPU instances
- Recommended instance types: g4dn.xlarge, g4dn.2xlarge
- Authentication through AWS credentials file or environment variables

### Google Cloud Platform
- Uses Compute Engine with GPU
- Recommended instance types: n1-standard-4 with NVIDIA T4
- Authentication through Google Cloud SDK

### Azure
- Uses Azure VMs with GPU
- Recommended instance types: Standard_NC6s_v3
- Authentication through Azure CLI

## Troubleshooting

### Common Issues

- **Missing DICOM files**: Ensure your DICOM directory contains valid DICOM files
- **GPU memory errors**: Reduce the number of points or use a cloud GPU with more memory
- **Web viewer not loading**: Check that all files were properly copied to the web output directory

### Cloud GPU Issues

- **Authentication errors**: Ensure you have valid credentials for the chosen cloud provider
- **Instance limits**: Check if you have reached instance limits for your cloud account
- **Network errors**: Ensure you have a stable internet connection for uploading/downloading data

## License

MIT License - See LICENSE file for details 