import numpy as np
import json
from scipy.spatial import KDTree
import trimesh


def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)


def crop_points_to_bbox(points, bb):
    # Create a boolean mask for points within the bounding box
    mask = np.logical_and(
        np.all(points >= bb[0], axis=1), np.all(points <= bb[1], axis=1)
    )

    # Apply the mask to the points array to get the cropped points
    cropped_points = points[mask]

    return cropped_points


def compute_transformation_error(t1, t2):
    eps = 1e-6

    # Rotation
    r1 = t1[:3, :3]
    r2 = t2[:3, :3]
    rot_error = (np.trace(r1 @ r2.T) - 1) / 2
    rot_error = np.clip(rot_error, -1.0 + eps, 1.0 - eps)
    rot_error = np.arccos(rot_error)

    # Translation
    tr1 = t1[:3, 3]
    tr2 = t2[:3, 3]
    tr_error = np.linalg.norm(tr1 - tr2)

    return (rot_error, tr_error)


def pose_estimate(d1, d2, scale):
    total_error_rotation = 0.0
    total_error_translation = 0.0

    keys = d1.keys()

    for camera in keys:
        transform1 = np.array(d1[camera], dtype=np.float32)
        transform2 = np.array(d2[camera], dtype=np.float32)
        transform2 = np.vstack((transform2, np.array([0, 0, 0, 1], dtype=np.float32)))
        transform2[:3, 3] /= scale
        transform2 = transform2

        rot_err, tr_err = compute_transformation_error(transform1, transform2)

        total_error_rotation += rot_err
        total_error_translation += tr_err

    total_error_rotation /= len(keys)
    total_error_translation /= len(keys)

    return total_error_rotation, total_error_translation

