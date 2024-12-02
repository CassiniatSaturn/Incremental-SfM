import trimesh
import numpy as np
import open3d as o3d

test_frame = 51
item = "boot"
pcd_pred = trimesh.load(f"estimation/{item}_test{test_frame}_opt.ply")
# pcd_pred = trimesh.load(f"estimation/tri_bug.ply")
pred_points = np.array(pcd_pred.vertices, dtype=np.float32)
pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_points))
pcd2.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_plotly(
    [pcd2],
    width=1980,
    height=1080,
)
