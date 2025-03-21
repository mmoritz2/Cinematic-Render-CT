# Cinematic Render CT

A tool for creating cinematic 3D visualizations from CT scan DICOM data, using 3D Gaussian Splatting techniques.

## Features

- Convert DICOM CT scans to 3D point clouds
- Train 3D Gaussian models for high-quality volumetric rendering
- Interactive web-based viewing of CT scan data
- Support for both CUDA (NVIDIA) and Metal (Apple) GPU acceleration

## Requirements

- Python 3.8+
- PyTorch
- DICOM processing libraries (pydicom, etc.)
- Either:
  - NVIDIA GPU with CUDA support, or
  - Apple Silicon/Intel Mac with Metal support (using gsplat)

## Installation

Clone the repository:

```bash
git clone https://github.com/mmoritz2/Cinematic-Render-CT.git
cd Cinematic-Render-CT
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For Apple Silicon/Intel Mac users, also install gsplat:

```bash
pip install gsplat
```

## Usage

### Basic Usage

```bash
cd ct-cinematic-viewer
python main.py --dicom_dir "path/to/dicom" --output_dir "../output"
```

### Using with Apple Silicon/Intel Mac (Metal)

```bash
cd ct-cinematic-viewer
python main.py --dicom_dir "path/to/dicom" --output_dir "../output" --use_gsplat
```

### Options

- `--dicom_dir`: Directory containing DICOM files
- `--output_dir`: Output directory for generated files
- `--threshold`: HU threshold for segmentation (default: 250)
- `--num_points`: Number of points to sample (default: 100000)
- `--num_cameras`: Number of camera views (default: 50)
- `--iterations`: Number of training iterations (default: 30000)
- `--kernel_size`: Initial kernel size for Gaussians (default: 0.1)
- `--skip_dicom`: Skip DICOM processing step
- `--skip_training`: Skip training and compression
- `--use_gsplat`: Use gsplat (Metal) instead of CUDA
- `--serve`: Start a web server after preparation
- `--port`: Port for the web server (default: 8000)

## Viewing Results

After running the pipeline, you can view the results by:

1. Using the built-in web server (with `--serve` flag)
2. Opening `output/web/index.html` in a browser
3. Starting a web server manually:

```bash
cd output
python -m http.server 8000
```

Then visit http://localhost:8000/web/ in your browser.

## Structure

- `ct-cinematic-viewer/`: Main code directory
  - `main.py`: Entry point and pipeline coordinator
  - `dicom_processor.py`: DICOM to point cloud conversion
  - `train_compress.py`: CUDA-based training and compression
  - `gsplat_train.py`: Metal-based training using gsplat
  - `train_gsplat.py`: Pipeline for gsplat training
- `cinematic-gaussians/`: 3D Gaussian model implementation
- `web-splat/`: Web viewer for 3D models

## License

[MIT License](LICENSE)

## Acknowledgements

This project builds upon:
- 3D Gaussian Splatting research
- gsplat for Metal acceleration
- Various open-source DICOM processing tools 