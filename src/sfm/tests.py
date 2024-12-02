import json
import numpy as np

from view_graph import ViewGraph
from types import SimpleNamespace
import cv2 as cv
from utils import save_point_to_ply, read_correspondences
import numpy as np
import json
from scipy.spatial import KDTree
import trimesh
import os
import torch


# read view graph from json
# return a dictionary of view graph
def read_view_graph(path: str) -> dict:
    with open(path, "r") as f:
        # view_graph = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        # view_graph = json.load(f, object_hook=lambda d: ViewGraph(**d))
        view_graph = json.load(f)
    return view_graph


test_frame = 10
item = "milk"

view_graph = read_view_graph(f"estimation/vg_{test_frame}_opt.json")
gt_camera_param = json.load(open("data/milk/gt_camera_parameters.json"))["extrinsics"]
pred_camera_param = json.load(open(f"estimation/milk_test{test_frame}_opt.json"))[
    "extrinsics"
]
K = np.array(json.load(open("data/box/gt_camera_parameters.json"))["intrinsics"])


def get_gt_camera_incremental(
    gt_camera_param, pred_camera_param, test_frame: int
) -> dict:
    keys = list(pred_camera_param.keys())
    gt_camera_param = {k: gt_camera_param[k] for k in keys[:test_frame]}
    gt_camera_param = {"extrinsics": gt_camera_param}
    json.dump(gt_camera_param, open("data/milk/gt_camera_parameters2.json", "w"))
    return gt_camera_param


get_gt_camera_incremental(gt_camera_param, pred_camera_param, test_frame)


def triangulate_points_3d(view_graph: dict, gt_camera_param: dict) -> np.ndarray:
    """
    Triangulate 3D points from view graph and ground truth camera parameters
    """

    acc_error = 0
    gt_points = []
    pred_points = []
    gt_cameras = dict()
    for node in view_graph["nodes"].values():
        # pick the first two cameras to triangulate points
        subnodes_id = list(node["subnodes"].keys())
        if len(subnodes_id) < 2:
            continue
        camera1 = node["subnodes"][subnodes_id[0]]["frame_id"]
        camera2 = node["subnodes"][subnodes_id[1]]["frame_id"]
        camera1_key = f"{camera1}".zfill(5) + ".jpg"
        camera2_key = f"{camera2}".zfill(5) + ".jpg"
        camera1_pose = np.array(gt_camera_param[camera1_key])
        camera2_pose = np.array(gt_camera_param[camera2_key])
        gt_cameras[camera1_key] = camera1_pose
        gt_cameras[camera2_key] = camera2_pose
        projection_matrix1 = K @ camera1_pose[:3, :]
        projection_matrix2 = K @ camera2_pose[:3, :]
        point_gt = cv.triangulatePoints(
            projection_matrix1,
            projection_matrix2,
            np.array(node["subnodes"][subnodes_id[0]]["position"]),
            np.array(node["subnodes"][subnodes_id[1]]["position"]).squeeze(),
        )

        point_gt = point_gt / point_gt[3]
        point_gt = point_gt.squeeze()

        error = np.linalg.norm(node["position"] - point_gt)
        diff = abs(node["position"] - point_gt)
        gt_points.append(point_gt)
        pred_points.append(node["position"])
        acc_error += error

    save_point_to_ply(np.array(gt_points), f"gt_points{test_frame}.ply")


# triangulate_points_3d(view_graph, gt_camera_param)


def check_corr2d(view_graph: dict) -> np.ndarray:
    """
    Triangulate 3D points from view graph and ground truth camera parameters
    """

    gt_cameras = dict()
    corr_2d = dict()

    gt_corr, gt_corr_keys = read_correspondences("data/box/correspondences")
    unmatched = []

    for node in view_graph["nodes"].values():
        # pick the first two cameras to triangulate points
        subnodes_id = list(node["subnodes"].keys())
        point_2d_frames = []
        point_2d_positions = []

        for subnode_id, subnode in node["subnodes"].items():
            position = subnode["position"]
            frame_id = subnode["frame_id"]
            point_2d_frames.append(int(frame_id))
            point_2d_positions.append(position)

        # get corr pairs
        # Generate all 2-element combinations
        combinations_list = []
        for i in range(len(point_2d_frames)):
            for j in range(i + 1, len(point_2d_frames)):
                if point_2d_frames[i] < point_2d_frames[j]:
                    corr_pair = f"{point_2d_frames[i]}_{point_2d_frames[j]}"
                    # check corr_pair is in the correspondence list
                    assert corr_pair in gt_corr_keys
                    combinations_list.append(corr_pair)
                else:
                    assert corr_pair in gt_corr_keys
                    corr_pair = f"{point_2d_frames[j]}_{point_2d_frames[i]}"
                    combinations_list.append(corr_pair)

        for i, comb in enumerate(combinations_list):
            k1, k2 = comb.split("_")
            match = torch.stack(
                (
                    torch.tensor(
                        point_2d_positions[point_2d_frames.index(int(k1))]
                    ).squeeze(),
                    torch.tensor(
                        (point_2d_positions[point_2d_frames.index(int(k2))]),
                    ).squeeze(),
                )
            ).flatten()

            # check if the key is in the dict
            check = match in gt_corr[comb]
            if not check:
                unmatched.append({comb: match})

    if len(unmatched) > 0:
        print("Unmatched pairs: ", len(unmatched))
    else:
        print("All pairs are matched")


def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)


# compute chamfer distance from gt points and estimated points
# check_corr2d(view_graph)
# gt_points = trimesh.load("estimation/gt_points.ply")
# gt_points = np.array(gt_points.vertices, dtype=np.float32)
# estimated_points = trimesh.load(f"estimation/box_test{test_frame}.ply")
# estimated_points = np.array(estimated_points.vertices, dtype=np.float32)
# chamfer_dist = chamfer_distance(gt_points, estimated_points)
# print(f"Chamfer distance: {chamfer_dist}")
