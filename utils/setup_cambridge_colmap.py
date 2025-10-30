import math
import os

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# setup individual scene IDs and their download location
scenes = [
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
    "GreatCourt"
]

target_height = 480  # rescale images

root = r"data/Cambridge"
os.makedirs(root, exist_ok=True)

for scene in scenes:
    print(f"Processing {scene}")

    print("Loading SfM reconstruction...")
    datadir = f'{root}/{scene}'
    input_file = f'{datadir}/reconstruction.nvm'
    valid_images = set()
    with open(input_file) as f:
        f.readline()
        f.readline()
        n_images = int(f.readline().strip())
        print(f"Number of images: {n_images}")
        for i in range(n_images):
            image_path, *_ = f.readline().strip().split()
            valid_images.add(image_path)
    if scene == 'ShopFacade':
        valid_images.remove("seq2/frame00036.jpg")
        valid_images.remove("seq2/frame00043.jpg")
    modes = ['train', 'test']
    img_output_folder = f'{datadir}/colmap/images'
    pose_output_folder = f'{datadir}/colmap/sparse/0'
    test_list_output_folder = f'{datadir}/colmap/undistorted'
    os.makedirs(img_output_folder, exist_ok=True)
    os.makedirs(pose_output_folder, exist_ok=True)
    os.makedirs(test_list_output_folder, exist_ok=True)
    poses = []
    with open(f"{test_list_output_folder}/list_test.txt", "w") as test_output:
        for mode in modes:
            print(f"Converting {mode} data...")
            # get list of images for current mode (train vs. test)
            image_list = f'{datadir}/dataset_{mode}.txt'
            with open(image_list) as f:
                lines = f.readlines()
            for line in tqdm(lines[3:]):
                line = line.strip()
                if not line:
                    continue
                image_path, *pose = line.strip().split()
                image_name = image_path.replace(".png", ".jpg")
                if image_name not in valid_images:
                    print(f"Skipping image {image_name}. Not part of reconstruction.")
                    continue
                image_name = image_name.replace('/', '_')
                if mode == 'test':
                    test_output.write(f"{image_name}\n")

                tx, ty, tz, qw, qx, qy, qz = map(float, pose)
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                extr = np.eye(4)
                extr[:3, :3] = R.T
                extr[:3, 3] = [tx, ty, tz]
                if np.absolute(extr[:3, 3]).max() > 10000:
                    print(f"Skipping image {image_name}. Extremely large translation. Outlier?")
                    print(extr[:3, 3])
                    continue
                pose = np.linalg.inv(extr)
                tx, ty, tz = pose[:3, 3]
                qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
                poses.append([image_name, qw, qx, qy, qz, tx, ty, tz])

                image = cv.imread(f"{datadir}/{image_path}")
                img_aspect = image.shape[0] / image.shape[1]
                if img_aspect > 1:
                    # portrait
                    img_w = target_height
                    img_h = int(math.ceil(target_height * img_aspect))
                else:
                    # landscape
                    img_w = int(math.ceil(target_height / img_aspect))
                    img_h = target_height
                image = cv.resize(image, (img_w, img_h))
                cv.imwrite(f"{img_output_folder}/{image_name}", image, [cv.IMWRITE_JPEG_QUALITY, 100])
    
    # Get camera intrinsics from reconstruction.nvm file
    # NVM format: <Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
    focal_length = None
    with open(input_file) as f:
        lines = f.readlines()
        # Skip header lines
        if len(lines) > 2:
            # Look for camera parameters in the image list
            for i in range(3, min(10, len(lines))):  # Check first few images
                parts = lines[i].strip().split()
                if len(parts) >= 5:
                    try:
                        focal_length = float(parts[1])
                        break
                    except (ValueError, IndexError):
                        continue
    
    # If no focal length found in NVM, estimate from image dimensions
    if focal_length is None:
        # Try to read a sample image to get dimensions
        sample_img_path = None
        for pose in poses[:1]:
            if pose:
                sample_img_path = f"{img_output_folder}/{pose[0]}"
                break
        if sample_img_path and os.path.exists(sample_img_path):
            sample_img = cv.imread(sample_img_path)
            if sample_img is not None:
                img_h, img_w = sample_img.shape[:2]
                # Estimate focal length (typical for Cambridge: ~525 pixels for 640x480)
                focal_length = img_w * 0.82  # Approximate scaling factor
                cx, cy = img_w / 2, img_h / 2
            else:
                img_w, img_h = 640, 480
                focal_length = 525.0
                cx, cy = img_w / 2, img_h / 2
        else:
            # Fallback values
            img_w, img_h = 640, 480
            focal_length = 525.0
            cx, cy = img_w / 2, img_h / 2
    else:
        # Use the focal length from NVM, but need to scale if images were resized
        # Original Cambridge images are typically 640x480, we resize to target_height=480
        # So we need to scale the focal length proportionally
        sample_img_path = None
        for pose in poses[:1]:
            if pose:
                sample_img_path = f"{img_output_folder}/{pose[0]}"
                break
        if sample_img_path and os.path.exists(sample_img_path):
            sample_img = cv.imread(sample_img_path)
            if sample_img is not None:
                img_h, img_w = sample_img.shape[:2]
                # Scale focal length from original to resized dimensions
                # Original height was ~480, new height is img_h
                focal_length = focal_length * (img_h / 480.0)
                cx, cy = img_w / 2, img_h / 2
            else:
                img_w, img_h = 640, 480
                focal_length = 525.0
                cx, cy = img_w / 2, img_h / 2
        else:
            img_w, img_h = 640, 480
            focal_length = 525.0
            cx, cy = img_w / 2, img_h / 2
    
    # Write cameras.txt (PINHOLE format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[])
    with open(f"{pose_output_folder}/cameras.txt", "w") as cam_output:
        cam_output.write("# Camera list with one line of data per camera:\n")
        cam_output.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        cam_output.write(f"1 PINHOLE {img_w} {img_h} {focal_length} {focal_length} {cx} {cy}\n")
    
    # Write points3D.txt (empty or minimal - COLMAP format requires this file)
    with open(f"{pose_output_folder}/points3D.txt", "w") as points_output:
        points_output.write("# 3D point list with one line of data per point:\n")
        points_output.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        # Empty point cloud - will be generated during 3DGS training
    
    poses.sort(key=lambda x: x[0])
    with open(f"{pose_output_folder}/images.txt", "w") as pose_output:
        pose_output.write("# Image list with two lines of data per image:\n")
        pose_output.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        pose_output.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, pose in enumerate(poses):
            image_name, qw, qx, qy, qz, tx, ty, tz = pose
            pose_output.write(f"{i + 1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {image_name}\n\n")
