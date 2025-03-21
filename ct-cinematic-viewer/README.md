# CT Cinematic Viewer

A pipeline to view CT images using cinematic rendering based on 3D Gaussian Splatting. This tool takes DICOM files as input and produces an interactive 3D visualization viewable in a web browser.

## Overview

This pipeline consists of several steps:

1. **DICOM Processing**: CT scan DICOM files are loaded, converted to Hounsfield units, and segmented using thresholding to create a surface mesh. The mesh is then sampled to create a point cloud.

2. **3D Gaussian Training**: The point cloud is used to train a 3D Gaussian Splatting model, which represents the CT data as a set of 3D Gaussians.

3. **Model Compression**: The trained model is compressed for efficient web viewing.

4. **Web Visualization**: The compressed model is rendered in a web browser using WebGPU, allowing for interactive exploration of the CT data.

## Prerequisites

- Python 3.9+
- Conda (for environment management)
- DICOM files from a CT scan

## Dependencies

This project depends on the following repositories:
- [cinematic-gaussians](https://github.com/KeKsBoTer/cinematic-gaussians)
- [web-splat](https://github.com/KeKsBoTer/web-splat)

They will be cloned automatically if not already present.

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/ct-cinematic-viewer.git
cd ct-cinematic-viewer
```

2. Install Python dependencies:
```
pip install -r requirements.txt
```

## Usage

### Basic Usage

```
python main.py --dicom_dir /path/to/dicom/files --output_dir /path/to/output/directory --serve
```

This will:
1. Process the DICOM files
2. Train a 3D Gaussian model
3. Compress the model
4. Set up a web viewer
5. Start a local web server to view the result

### Command Line Arguments

- `--dicom_dir`: Directory containing DICOM files (required)
- `--output_dir`: Base output directory (required)
- `--threshold`: HU threshold for segmentation (default: 250)
- `--num_points`: Number of points to sample from mesh (default: 100000)
- `--num_cameras`: Number of virtual cameras for training (default: 50)
- `--iterations`: Number of training iterations (default: 30000)
- `--skip_dicom`: Skip DICOM processing (use when you already have processed data)
- `--skip_training`: Skip training and compression (use when you already have a trained model)
- `--serve`: Start a web server after preparation
- `--port`: Port for the web server (default: 8000)

### Examples

#### Process CT data but don't start server:
```
python main.py --dicom_dir /path/to/dicom/files --output_dir /path/to/output/directory
```

#### Skip DICOM processing if you've already done it:
```
python main.py --dicom_dir /path/to/dicom/files --output_dir /path/to/output/directory --skip_dicom --serve
```

#### Skip training if you've already trained a model:
```
python main.py --dicom_dir /path/to/dicom/files --output_dir /path/to/output/directory --skip_training --serve
```

## Viewing the Results

If you used the `--serve` option, the viewer will be available at http://localhost:8000 (or the port you specified).

Otherwise, you can:
1. Navigate to the web output directory: `cd /path/to/output/directory/web`
2. Start a web server: `python -m http.server 8000`
3. Open a browser and go to http://localhost:8000

## Acknowledgements

This project builds upon the following research:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Cinematic Gaussians](https://keksboter.github.io/cinematic-gaussians/)
- [Compressed 3D Gaussian Splatting](https://github.com/KeKsBoTer/c3dgs)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 