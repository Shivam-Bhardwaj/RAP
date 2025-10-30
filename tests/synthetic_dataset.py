"""
Utility to create a minimal synthetic dataset from an existing dataset for testing.

This allows us to create a small subset of an existing dataset that can be used
for end-to-end testing without requiring the full dataset.
"""
import os
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import torch
from PIL import Image


def create_synthetic_dataset(
    source_dataset_path: str,
    output_path: str,
    num_train_images: int = 10,
    num_test_images: int = 5,
    seed: int = 42
) -> Tuple[str, str]:
    """
    Create a minimal synthetic dataset from an existing dataset.
    
    Args:
        source_dataset_path: Path to the source dataset (should have sparse/0/ and images/)
        output_path: Path where synthetic dataset will be created
        num_train_images: Number of training images to include
        num_test_images: Number of test images to include
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (synthetic_dataset_path, synthetic_model_path)
    """
    np.random.seed(seed)
    source_path = Path(source_dataset_path)
    output_path = Path(output_path)
    
    # Validate source dataset structure
    if not (source_path / "sparse/0").exists():
        raise ValueError(f"Source dataset must have sparse/0/ directory: {source_path}")
    if not (source_path / "images").exists():
        raise ValueError(f"Source dataset must have images/ directory: {source_path}")
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    sparse_dir = output_path / "sparse/0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Read original colmap files
    cameras_txt = source_path / "sparse/0/cameras.txt"
    images_txt = source_path / "sparse/0/images.txt"
    points3d_txt = source_path / "sparse/0/points3D.txt"
    
    cameras_bin = source_path / "sparse/0/cameras.bin"
    images_bin = source_path / "sparse/0/images.bin"
    points3d_bin = source_path / "sparse/0/points3D.bin"
    
    # Check if binary or text format
    use_binary = cameras_bin.exists() and images_bin.exists() and points3d_bin.exists()
    
    if use_binary:
        # For binary format, we'll need to read and write as text for simplicity
        # This is a simplified approach - for full binary support, use pycolmap
        print("Warning: Binary COLMAP format detected. Converting to text format for synthetic dataset.")
        use_binary = False
    
    if not cameras_txt.exists() and not cameras_bin.exists():
        raise ValueError(f"No camera files found in {source_path}/sparse/0/")
    
    # Read cameras
    if cameras_txt.exists():
        cameras = _read_cameras_txt(cameras_txt)
    else:
        # Fallback: create minimal camera
        cameras = {1: {
            'model': 'PINHOLE',
            'width': 640,
            'height': 480,
            'params': [500.0, 500.0, 320.0, 240.0]
        }}
    
    # Read images
    if images_txt.exists():
        images = _read_images_txt(images_txt)
    else:
        raise ValueError(f"No images.txt found. Cannot create synthetic dataset.")
    
    # Filter images by train/test split if available
    test_list_file = source_path / "list_test.txt"
    test_images_set = set()
    if test_list_file.exists():
        with open(test_list_file, 'r') as f:
            test_images_set = set(line.strip() for line in f)
    
    # Separate train and test images
    train_images = []
    test_images = []
    for img_id, img_data in images.items():
        if img_data['name'] in test_images_set:
            test_images.append((img_id, img_data))
        else:
            train_images.append((img_id, img_data))
    
    # If no explicit split, use first N as test
    if len(test_images) == 0:
        total_images = list(images.items())
        test_images = total_images[:num_test_images]
        train_images = total_images[num_test_images:num_test_images + num_train_images]
    
    # Sample train and test images
    if len(train_images) > num_train_images:
        train_images = np.random.choice(len(train_images), num_train_images, replace=False)
        train_images = [list(images.items())[i] for i in train_images]
    else:
        train_images = train_images[:num_train_images]
    
    if len(test_images) > num_test_images:
        test_images = np.random.choice(len(test_images), num_test_images, replace=False)
        test_images = [list(images.items())[i] for i in test_images]
    else:
        test_images = test_images[:num_test_images]
    
    # Collect all selected images
    selected_images = {img_id: img_data for img_id, img_data in train_images + test_images}
    selected_image_ids = set(selected_images.keys())
    
    # Copy selected images
    print(f"Copying {len(selected_images)} images...")
    for img_id, img_data in selected_images.items():
        src_img_path = source_path / "images" / img_data['name']
        if src_img_path.exists():
            dst_img_path = images_dir / img_data['name']
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image not found: {src_img_path}")
    
    # Write cameras.txt (use first camera)
    first_cam_id = list(cameras.keys())[0]
    first_cam = cameras[first_cam_id]
    with open(sparse_dir / "cameras.txt", 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"{first_cam_id} {first_cam['model']} {first_cam['width']} {first_cam['height']} ")
        f.write(" ".join(map(str, first_cam['params'])) + "\n")
    
    # Write images.txt
    with open(sparse_dir / "images.txt", 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(selected_images)}\n")
        
        for img_id, img_data in selected_images.items():
            qw, qx, qy, qz = img_data['quaternion']
            tx, ty, tz = img_data['translation']
            cam_id = img_data['camera_id']
            name = img_data['name']
            
            # Write image line
            f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {name}\n")
            
            # Write empty points2D line (simplified)
            f.write(" \n")
    
    # Write minimal points3D.txt
    with open(sparse_dir / "points3D.txt", 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")
    
    # Create list_test.txt for train/test split
    test_names = [img_data['name'] for _, img_data in test_images]
    with open(output_path / "list_test.txt", 'w') as f:
        for name in test_names:
            f.write(f"{name}\n")
    
    # Create model_path directory and cameras.json
    model_path = output_path / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Generate cameras.json from selected images
    cameras_json = []
    for idx, (img_id, img_data) in enumerate(selected_images.items()):
        # Convert quaternion to rotation matrix
        qw, qx, qy, qz = img_data['quaternion']
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        
        # Get translation
        tx, ty, tz = img_data['translation']
        T = np.array([tx, ty, tz])
        
        # Create world-to-camera matrix
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        
        # Convert to camera-to-world
        c2w = np.linalg.inv(w2c)
        position = c2w[:3, 3]
        rotation = c2w[:3, :3]
        
        # Get camera intrinsics
        cam = cameras[img_data['camera_id']]
        fx = cam['params'][0]
        fy = cam['params'][1] if len(cam['params']) > 1 else cam['params'][0]
        
        camera_entry = {
            'id': idx,
            'img_name': img_data['name'],
            'width': cam['width'],
            'height': cam['height'],
            'position': position.tolist(),
            'rotation': rotation.tolist(),
            'fy': fy,
            'fx': fx
        }
        cameras_json.append(camera_entry)
    
    # Save cameras.json
    with open(model_path / "cameras.json", 'w') as f:
        json.dump(cameras_json, f, indent=2)
    
    print(f"Synthetic dataset created at: {output_path}")
    print(f"  - {len(train_images)} training images")
    print(f"  - {len(test_images)} test images")
    print(f"  - Model path: {model_path}")
    
    return str(output_path), str(model_path)


def _read_cameras_txt(cameras_file: Path) -> dict:
    """Read cameras.txt file."""
    cameras = {}
    with open(cameras_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        
        parts = line.split()
        if len(parts) < 5:
            continue
        
        cam_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(p) for p in parts[4:]]
        
        cameras[cam_id] = {
            'model': model,
            'width': width,
            'height': height,
            'params': params
        }
    
    return cameras


def _read_images_txt(images_file: Path) -> dict:
    """Read images.txt file."""
    images = {}
    with open(images_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue
        
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]
        
        images[img_id] = {
            'quaternion': (qw, qx, qy, qz),
            'translation': (tx, ty, tz),
            'camera_id': cam_id,
            'name': name
        }
        
        i += 2  # Skip points2D line
    
    return images


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    # Normalize quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create synthetic dataset from existing dataset")
    parser.add_argument("--source", type=str, required=True, help="Source dataset path")
    parser.add_argument("--output", type=str, required=True, help="Output path for synthetic dataset")
    parser.add_argument("--num_train", type=int, default=10, help="Number of training images")
    parser.add_argument("--num_test", type=int, default=5, help="Number of test images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    create_synthetic_dataset(
        args.source,
        args.output,
        num_train_images=args.num_train,
        num_test_images=args.num_test,
        seed=args.seed
    )

