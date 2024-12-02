import numpy as np
from camera_pose_visualizer import CameraPoseVisualizer
import json
import matplotlib as plt
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--path", type=str, default="estimation/box_incremental_poses.json"
)
argparser.add_argument("--figtitle", type=str, default="estimation")


def visualize_pose(data_path: str, title: str):
    camera_param = json.load(open(data_path))
    # argument : the minimum/maximum value of x, y, z
    visualizer = CameraPoseVisualizer([-10, 10], [-10, 10], [-10, 10])
    poses = camera_param["extrinsics"]

    colormap = plt.colormaps["rainbow"]
    for pose in poses.values():
        pose = np.array(pose)
        visualizer.extrinsic2pyramid(pose, colormap(np.random.rand()), 1)

    visualizer.show(title)


if __name__ == "__main__":
    np.random.seed(0)
    args = argparser.parse_args()
    visualize_pose(args.path, args.figtitle)
