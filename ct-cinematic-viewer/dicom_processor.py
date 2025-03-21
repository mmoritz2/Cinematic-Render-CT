import os
import sys
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import SimpleITK as sitk
from pathlib import Path
import argparse
import json
from skimage import measure
import trimesh

def load_scan(path):
    """Load DICOM files from a directory."""
    print(f"Loading DICOM files from {path}")
    slices = []
    for s in os.listdir(path):
        if not s.lower().endswith('.dcm'):
            continue
        try:
            slices.append(pydicom.dcmread(os.path.join(path, s)))
        except Exception as e:
            print(f"Error reading {s}: {e}")
    
    # Sort slices by ImagePositionPatient
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    return slices

def get_pixels_hu(scans):
    """Convert DICOM to Hounsfield units."""
    images = np.stack([s.pixel_array for s in scans])
    
    # Convert to HU
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    images = images.astype(np.int16)
    images[images == -2000] = 0
    
    if slope != 1:
        images = slope * images.astype(np.float64)
        images = images.astype(np.int16)
        
    images += np.int16(intercept)
    return images

def resample_volume(image, scan, new_spacing=[1,1,1]):
    """Resample the volume to have isotropic voxels."""
    # Get current spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)
    
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor)
    real_resize = new_shape / image.shape
    new_spacing = spacing / real_resize
    
    image = sitk.GetImageFromArray(image)
    # The array is 3D but resample_sitk takes a sequence of factors
    # one for each dimension, so we need to provide them
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(new_spacing.tolist())
    resample_filter.SetSize(new_shape.astype(np.int32).tolist())
    resample_filter.SetInterpolator(sitk.sitkLinear)
    resample_filter.SetOutputDirection([1,0,0,0,1,0,0,0,1])
    resample_filter.SetOutputOrigin([0,0,0])
    resample_filter.SetDefaultPixelValue(-1000)
    
    resampled_image = resample_filter.Execute(image)
    return sitk.GetArrayFromImage(resampled_image)

def threshold_segment(image, min_threshold, max_threshold=3000):
    """Threshold the image to segment structures."""
    return np.logical_and(image >= min_threshold, image <= max_threshold)

def create_mesh_from_segmentation(segmentation, spacing):
    """Create a mesh from a binary segmentation using marching cubes."""
    # Use marching cubes to create a mesh
    verts, faces, normals, values = measure.marching_cubes(segmentation, 0)
    
    # Scale vertices by spacing
    verts = verts * spacing
    
    # Create a mesh from vertices and faces
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh

def mesh_to_pointcloud(mesh, num_points=100000):
    """Sample points from mesh to create a point cloud."""
    # Sample points from the mesh surface
    points = mesh.sample(num_points)
    
    # Add random colors for visualization (can be adjusted later)
    colors = np.random.rand(num_points, 3)
    
    return points, colors

def save_pointcloud(points, colors, output_file):
    """Save point cloud to PLY format."""
    pc = np.hstack([points, colors])
    with open(output_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float red\n")
        f.write("property float green\n")
        f.write("property float blue\n")
        f.write("end_header\n")
        for p in pc:
            f.write(f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]}\n")
    
    print(f"Point cloud saved to {output_file}")

def create_camera_views(points, num_cameras=50, output_file="cameras.json"):
    """Generate camera views around the point cloud for training."""
    # Calculate centroid and extent of point cloud
    centroid = np.mean(points, axis=0)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    extent = np.linalg.norm(max_bound - min_bound)
    
    # Generate camera positions on a sphere around the centroid
    cameras = []
    radius = extent * 1.5  # Place cameras at 1.5x the extent
    
    for i in range(num_cameras):
        # Generate points on a sphere using fibonacci sequence for uniform distribution
        phi = (1 + np.sqrt(5)) / 2
        y = 1 - (i / float(num_cameras - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)
        theta = 2 * np.pi * i / phi
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        
        # Scale to desired radius and translate to centroid
        position = np.array([x, y, z]) * radius + centroid
        
        # Look at centroid
        look_at = centroid
        up = np.array([0, 1, 0])  # Assuming Y is up
        
        # Create camera entry in the format expected by the 3DGS code
        camera = {
            "id": i,
            "img_name": f"camera_{i:03d}",
            "width": 800,
            "height": 800,
            "position": position.tolist(),
            "rotation": calculate_rotation(position, look_at, up).tolist(),
            "fov": 60.0,
            "near": 0.1,
            "far": extent * 3
        }
        cameras.append(camera)
    
    # Save cameras to JSON
    with open(output_file, 'w') as f:
        json.dump(cameras, f, indent=2)
    
    print(f"Camera views saved to {output_file}")
    return cameras

def calculate_rotation(position, target, up):
    """Calculate rotation matrix for camera looking at target from position."""
    # Calculate forward vector (looking from position to target)
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    
    # Calculate right vector
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Recalculate up vector to ensure orthogonality
    up = np.cross(right, forward)
    
    # Create rotation matrix
    rotation = np.eye(4)
    rotation[:3, 0] = right
    rotation[:3, 1] = up
    rotation[:3, 2] = -forward  # Negate because cameras look along negative z
    
    return rotation

def process_dicom_to_pointcloud(dicom_dir, output_dir, threshold=250, num_points=100000, num_cameras=50):
    """Process DICOM files to create a point cloud and camera positions for 3DGS."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load DICOM files
    scans = load_scan(dicom_dir)
    
    # Convert to Hounsfield units
    volume = get_pixels_hu(scans)
    
    # Resample to isotropic voxels (1mm in each dimension)
    resampled_volume = resample_volume(volume, scans)
    
    # Get spacing information
    spacing = np.array([1.0, 1.0, 1.0])  # Our resampled spacing
    
    # Segment by thresholding
    segmentation = threshold_segment(resampled_volume, threshold)
    
    # Create mesh from segmentation
    mesh = create_mesh_from_segmentation(segmentation, spacing)
    
    # Create point cloud from mesh
    points, colors = mesh_to_pointcloud(mesh, num_points)
    
    # Save point cloud to PLY
    point_cloud_file = os.path.join(output_dir, "point_cloud.ply")
    save_pointcloud(points, colors, point_cloud_file)
    
    # Create camera views around the point cloud
    camera_file = os.path.join(output_dir, "cameras.json")
    create_camera_views(points, num_cameras, camera_file)
    
    return point_cloud_file, camera_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DICOM files for 3D Gaussian Splatting")
    parser.add_argument("--dicom_dir", required=True, help="Directory containing DICOM files")
    parser.add_argument("--output_dir", required=True, help="Output directory for point cloud and cameras")
    parser.add_argument("--threshold", type=int, default=250, help="HU threshold for segmentation")
    parser.add_argument("--num_points", type=int, default=100000, help="Number of points to sample")
    parser.add_argument("--num_cameras", type=int, default=50, help="Number of camera views")
    
    args = parser.parse_args()
    
    process_dicom_to_pointcloud(
        args.dicom_dir, 
        args.output_dir, 
        args.threshold, 
        args.num_points, 
        args.num_cameras
    ) 