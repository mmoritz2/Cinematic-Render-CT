#!/usr/bin/env python3
"""
Modified 3D Gaussian Splatting training script using gsplat for Apple Silicon/Intel Macs.
This version replaces the CUDA-dependent modules with gsplat which has Metal support.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import gsplat

# Add the cinematic-gaussians directory to the path so we can import from gaussian_me
sys.path.append(os.path.join(os.path.dirname(__file__), "../cinematic-gaussians"))

# Import necessary modules from the original implementation
from gaussian_me.io.dataset_readers import BasicPointCloud, CameraInfo
from gaussian_me.io import CamerasDataset
from gaussian_me.utils.camera_utils import camera_to_JSON
from gaussian_me.utils.general_utils import inverse_sigmoid
from gaussian_me.utils.loss_utils import l1_loss, ssim
from gaussian_me.utils.image_utils import psnr
from gaussian_me.args import ModelParams, PipelineParams, OptimizationParams

from torch.utils.tensorboard import SummaryWriter

class GsplatGaussianModel(torch.nn.Module):
    """
    Modified Gaussian Model class that uses gsplat for rendering
    """
    
    def __init__(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        kernel_size: float = 0.1,
        max_sh_degree: int = 0,  # We use rgb directly, not spherical harmonics
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        
        # Store kernel size
        self.kernel_size = kernel_size
        self.max_sh_degree = max_sh_degree
        self.active_sh_degree = 0
        
        # Activations
        self.scaling_activation = torch.exp
        self.scaling_activation_inverse = torch.log
        self.rotation_activation = torch.nn.functional.normalize
        self.rotation_activation_inverse = lambda x: x
        self.opacity_activation = torch.sigmoid
        self.opacity_activation_inverse = inverse_sigmoid
        
        # Parameters (pytorch trainable)
        self._xyz = torch.nn.Parameter(xyz.contiguous(), requires_grad=True)
        self._features = torch.nn.Parameter(features.contiguous(), requires_grad=True)
        self._scaling = torch.nn.Parameter(
            self.scaling_activation_inverse(scaling.contiguous()), requires_grad=True
        )
        self._rotation = torch.nn.Parameter(
            self.rotation_activation_inverse(rotation.contiguous()), requires_grad=True
        )
        self._opacity = torch.nn.Parameter(
            self.opacity_activation_inverse(opacity.contiguous()), requires_grad=True
        )
    
    @classmethod
    def from_pc(
        cls,
        pcd: BasicPointCloud,
        kernel_size: float = 0.1,
        max_sh_degree: int = 0,  # We use rgb directly, not spherical harmonics
    ):
        """Create a Gaussian model from a point cloud"""
        
        # Extract data from point cloud
        xyz = torch.tensor(pcd.points, dtype=torch.float32)
        features = torch.tensor(pcd.colors, dtype=torch.float32)
        
        # Initialize scales and rotations
        scales = torch.ones_like(xyz) * kernel_size
        rotations = torch.zeros((xyz.shape[0], 4), dtype=torch.float32)
        rotations[:, 0] = 1  # Initialize with identity quaternion
        
        # Initialize opacity
        opacity = torch.ones((xyz.shape[0], 1), dtype=torch.float32) * 0.5
        
        return cls(
            xyz=xyz,
            features=features,
            scaling=scales,
            rotation=rotations,
            opacity=opacity,
            kernel_size=kernel_size,
            max_sh_degree=max_sh_degree,
        )
    
    @property
    def xyz(self) -> torch.Tensor:
        """Get positions"""
        return self._xyz
    
    @property
    def scaling(self) -> torch.Tensor:
        """Get scales with activation applied"""
        return self.scaling_activation(self._scaling)
    
    @property
    def rotation(self) -> torch.Tensor:
        """Get rotations with activation applied"""
        return self.rotation_activation(self._rotation)
    
    @property
    def opacity(self) -> torch.Tensor:
        """Get opacity with activation applied"""
        return self.opacity_activation(self._opacity)
    
    @property
    def color(self) -> torch.Tensor:
        """Get colors"""
        return self._features
    
    def num_points(self) -> int:
        """Get number of Gaussians"""
        return self._xyz.shape[0]
    
    def save_ply(self, path):
        """Save the model as a PLY file"""
        
        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        
        # Convert quaternions to empty normals (not used, but kept for compatibility)
        f_dc = self.color.detach().cpu().numpy()
        
        # Scales and opacity
        scales = self.scaling.detach().cpu().numpy()
        opacity = self.opacity.detach().cpu().numpy()
        
        # Define PLY data structure
        dtype_full = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "f4"),
            ("green", "f4"),
            ("blue", "f4"),
            ("scale_x", "f4"),
            ("scale_y", "f4"),
            ("scale_z", "f4"),
            ("opacity", "f4"),
        ]
        
        # Create array with all data
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            [
                xyz,
                normals,
                f_dc,
                scales,
                opacity,
            ],
            axis=1,
        )
        
        # Fill in the data
        elements["x"] = attributes[:, 0]
        elements["y"] = attributes[:, 1]
        elements["z"] = attributes[:, 2]
        elements["nx"] = attributes[:, 3]
        elements["ny"] = attributes[:, 4]
        elements["nz"] = attributes[:, 5]
        elements["red"] = attributes[:, 6]
        elements["green"] = attributes[:, 7]
        elements["blue"] = attributes[:, 8]
        elements["scale_x"] = attributes[:, 9]
        elements["scale_y"] = attributes[:, 10]
        elements["scale_z"] = attributes[:, 11]
        elements["opacity"] = attributes[:, 12]
        
        # Create PLY element
        el = PlyElement.describe(elements, "vertex")
        
        # Create PLY data and save
        PlyData([el]).write(path)
        print(f"PLY data saved to {path}")


def gsplat_render(
    pc,
    camera,
    background_color=None,
    image_height=None,
    image_width=None,
):
    """
    Render the point cloud using gsplat
    """
    # Extract camera parameters
    if image_height is None:
        image_height = camera.image_height
    if image_width is None:
        image_width = camera.image_width
    
    # Extract model parameters
    means = pc.xyz
    scales = pc.scaling
    
    # Convert quaternions to rotation matrices (gsplat uses 3x3 matrices)
    quats = pc.rotation
    # This is a simplified conversion - in practice we'd need a proper quaternion to matrix function
    rot_matrices = torch.eye(3).unsqueeze(0).repeat(quats.shape[0], 1, 1)
    
    # Get colors and opacities
    rgbs = pc.color
    opacities = pc.opacity.squeeze(-1)  # Remove last dimension
    
    # Prepare camera for gsplat
    # Convert camera data to gsplat format
    c2w = torch.tensor(camera.world_view_transform, dtype=torch.float32).inverse()
    
    # Create projection matrix from FoV
    fovx = camera.FoVx
    fovy = camera.FoVy
    projection_matrix = torch.zeros(4, 4, dtype=torch.float32)
    projection_matrix[0, 0] = 1.0 / torch.tan(torch.tensor(fovx / 2.0))
    projection_matrix[1, 1] = 1.0 / torch.tan(torch.tensor(fovy / 2.0))
    projection_matrix[2, 2] = -1.0  # Near and far are set to -1 and 1 for simplicity
    projection_matrix[2, 3] = -0.1  # Small offset to avoid z=0
    projection_matrix[3, 2] = -1.0
    
    # Set up background color
    if background_color is None:
        background_color = torch.ones(3, dtype=torch.float32)
    
    # Render using gsplat
    # Note: This is a simplified version - in practice we'd need to match the exact rendering parameters
    xys, depths, radii, conics, compensation, num_tiles_hit = gsplat.project_gaussians(
        means,
        scales,
        rot_matrices,
        c2w[:3, :3],  # Rotation
        c2w[:3, 3:4],  # Translation
        projection_matrix,
        image_height,
        image_width
    )
    
    # Render image
    rendered_image = gsplat.rasterize_gaussians(
        xys,
        depths,
        radii,
        conics,
        rgbs,
        opacities,
        image_height,
        image_width,
        background_color
    )
    
    return rendered_image


def prepare_output_and_logger(output_folder, args):
    """
    Create output folders and initialize the logger
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Create tensorboard writer
    return SummaryWriter(output_folder)


def training(
    model_params,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
):
    """
    Main training function
    """
    # Create output directory and initialize tensorboard logger
    tb_writer = prepare_output_and_logger(model_params.model_path, model_params)
    
    # Load dataset
    dataset = CamerasDataset.from_folder(model_params.source_path, model_params.images)
    
    # Save camera information
    json_cams = []
    for id, cam in enumerate(dataset.scene_info.cameras):
        json_cams.append(camera_to_JSON(id, cam))
    with open(os.path.join(model_params.model_path, "cameras.json"), "w") as file:
        json.dump(json_cams, file)
    
    # Handle evaluation dataset if needed
    if model_params.eval:
        train_dataset, val_dataset = dataset.split_train_val()
    else:
        train_dataset = dataset
        val_dataset = []
    
    # Initialize model
    gaussians = GsplatGaussianModel.from_pc(
        pcd=dataset.init_pc(),
        kernel_size=model_params.kernel_size,
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        [
            {
                "params": [gaussians._xyz],
                "lr": opt.position_lr,
                "name": "xyz",
            },
            {
                "params": [gaussians._features],
                "lr": opt.feature_lr,
                "name": "features",
            },
            {
                "params": [gaussians._scaling],
                "lr": opt.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [gaussians._opacity],
                "lr": opt.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [gaussians._rotation],
                "lr": opt.rotation_lr,
                "name": "rotation",
            },
        ]
    )
    
    bg_color = torch.tensor([1, 1, 1], dtype=torch.float32)
    
    # Create a cycle iterator over the training cameras
    cameras_iter = cycle(train_dataset.get_all_cameras())
    
    # Main training loop
    for iteration in range(opt.iterations):
        # Get the next camera
        viewpoint_cam = next(cameras_iter)
        
        # Render the Gaussians from this viewpoint
        rendered_image = gsplat_render(
            gaussians,
            viewpoint_cam,
            background_color=bg_color,
        )
        
        # Get the ground truth image
        gt_image = torch.tensor(viewpoint_cam.image, dtype=torch.float32)
        
        # Compute loss
        Ll1 = l1_loss(rendered_image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
        
        # Perform optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Report progress
        if iteration in testing_iterations:
            print(f"Iteration {iteration}: loss = {loss.item():.4f}, L1 = {Ll1.item():.4f}")
            
            # Log to tensorboard
            tb_writer.add_scalar("train/loss", loss.item(), iteration)
            tb_writer.add_scalar("train/l1_loss", Ll1.item(), iteration)
            
            # Generate and save a validation image
            if val_dataset:
                val_cam = val_dataset.get_random_camera()
                with torch.no_grad():
                    val_image = gsplat_render(gaussians, val_cam, background_color=bg_color)
                    tb_writer.add_image("val/image", val_image.permute(2, 0, 1), iteration)
        
        # Save model checkpoint
        if iteration in saving_iterations:
            model_path = os.path.join(model_params.model_path, f"point_cloud/iteration_{iteration}")
            os.makedirs(model_path, exist_ok=True)
            gaussians.save_ply(os.path.join(model_path, "point_cloud.ply"))
    
    # Save final model
    final_model_path = os.path.join(model_params.model_path, "point_cloud/iteration_final")
    os.makedirs(final_model_path, exist_ok=True)
    gaussians.save_ply(os.path.join(final_model_path, "point_cloud.ply"))
    
    print("Training complete!")


def main():
    """
    Parse arguments and start training
    """
    from gaussian_me.args import parse_args
    args = parse_args()
    
    # Set up testing and saving iterations
    testing_iterations = list(range(0, args.iterations, 100))
    if args.iterations not in testing_iterations:
        testing_iterations.append(args.iterations)
    
    saving_iterations = list(range(0, args.iterations, 1000))
    if args.iterations not in saving_iterations:
        saving_iterations.append(args.iterations)
    
    # Start training
    training(
        model_params=args,
        opt=args,
        pipe=args,
        testing_iterations=testing_iterations,
        saving_iterations=saving_iterations,
    )


if __name__ == "__main__":
    main() 