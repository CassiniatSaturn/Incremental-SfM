import os
import json
import numpy as np
import torch
import cv2 as cv
import open3d as o3d


def project_points(X, R, t, K):
    """
    Project 3D points X onto the image plane using camera pose (R, t) and intrinsic matrix K.
    """
    # Convert R from vector to matrix if needed
    if R.shape == (3,):
        R, _ = cv.Rodrigues(R)
    # Project 3D points
    projected = R @ X.T + t
    projected = projected[:2] / projected[2]
    # Apply intrinsic parameters
    projected = K @ np.vstack((projected, np.ones((1, projected.shape[1]))))
    return projected[:2].T


def reprojection_error(
    params, n_cameras, n_points, camera_indices, point_indices, points_2d, K
):
    """
    Compute total reprojection error for all points in all cameras.
    """

    camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6 :].reshape((n_points, 3))
    error = []
    for i in range(points_2d.shape[0]):
        point_2d = points_2d[i]
        cam_idx = camera_indices[i]
        point_idx = point_indices[i]
        R, t = camera_params[cam_idx, :3], camera_params[cam_idx, 3:6]
        X = points_3d[point_idx]
        projected = project_points(X, R, t, K)
        error.append(projected - point_2d)
    return np.array(error).ravel()


"""
read correspondences from a text file
return:
    a dictionary of correspondences, the key is the frame1_id_frame2_id, and the value is a tensor of correspondences
    a list of frame to frame mapping
"""


def read_correspondences(dir: str) -> dict:
    # frame to frame correspondences: {frame1_id_frame2_id: correspondences}}
    correspondences = dict()
    f2f_mapping = []
    for file in sorted(os.listdir(dir)):
        f2f_ids, _ = file.split(".")

        with open(os.path.join(dir, file), "r") as input_file:
            data = np.loadtxt(
                input_file,
                dtype=np.float32,
            )
            assert len(data) > 0
            correspondences[f2f_ids] = torch.tensor(data)
            f2f_mapping.append(f2f_ids)

    return correspondences, f2f_mapping


"""
get correspondences by given frame1_id and frame2_id
return a tensor of correspondences
"""


def get_correspondences(correspondences: dict, f2f_ids: str) -> torch.tensor:
    return correspondences[f2f_ids]


"""
read grounth truth camera extrinstic parameters from a COLMAP estimated json file
return a dict in the format of {frame_id: transform_matrix} following GT format
"""


def get_gt_pose(path: str) -> dict:
    with open(path, "r") as f:
        gts = json.load(f)

    gts = gts["frames"]
    formatted_gts = dict()
    for gt in gts:
        file_path, M_ext = gt["file_path"], gt["transform_matrix"]
        formatted_gts[file_path] = M_ext
    formatted_gts = {"extrinsics": formatted_gts}
    return formatted_gts


"""
save a dictionary to a json file under the estimation folder
"""


def save_json(path: str, data: dict) -> None:
    file_path = os.path.join("./estimation", path)
    if not os.path.isfile(file_path):
        os.mknod(file_path)
    with open(file_path, "w") as f:
        json.dump(data, f)


"""
read camera intrinsics from a json file
return a tensor of camera intrinsics
"""


def get_intrinsics(path: str) -> torch.tensor:
    with open(path, "r") as f:
        camera_mats = json.load(f)
    K = camera_mats["intrinsics"]
    return torch.tensor(K)


"""
get frame ids from a f2f key in int or string format
return the tuple of frame1_id and frame2_id in integral or string format
"""


def get_frames_from_key(f2f_key: str, dtype="str") -> tuple:
    if dtype == "str":
        return f2f_key.split("_")[0], f2f_key.split("_")[1]
    if dtype == "int":
        return int(f2f_key.split("_")[0]), int(f2f_key.split("_")[1])


"""
    save camera poses to json file
 """


def save_camera_poses(poses: dict, file_name: str) -> None:
    # convert pose to nparray
    M_ext = dict()
    for key, value in poses.items():
        image_file = f"{key.zfill(5)}.jpg"
        M_ext[image_file] = value.tolist()
    file_path = os.path.join("./estimation", file_name)
    if not os.path.isfile(file_path):
        os.mknod(file_path)
    with open(file_path, "w") as f:
        format_M_ext = {"extrinsics": M_ext}
        json.dump(format_M_ext, f)


"""
    save sparse point cloud to ply file
"""


def save_point_to_ply(points: np.ndarray, file_name: str) -> None:
    _, dim = points.shape
    if dim == 4:
        points = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(os.path.join("./estimation/", file_name), pcd)


"""
    read sparse point cloud from ply file return n*3 np array
"""


def read_ply(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    return np.array(pcd.points, dtype=np.float32)


