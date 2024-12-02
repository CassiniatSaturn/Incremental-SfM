from utils import (
    read_correspondences,
    get_correspondences,
    get_intrinsics,
    reprojection_error,
    get_frames_from_key,
    save_camera_poses,
    save_point_to_ply,
)
import cv2 as cv
import torch
import numpy as np
from scipy.optimize import least_squares
from view_graph import ViewGraph
import os
from triangulate import linear_triangulation, normalize_pts


# Structure from Motion implemented using OpenCV and PyTorch
class SFM:
    def __init__(
        self,
        intrinsics: torch.tensor,
        n_views: int,
        images_dir: str,
        window_size=10,
        f2f_mapping: list = None,
        correspondences: dict = None,
    ):
        # Use SIFT as default feature detector and extractor, can be changed to D2Net
        self.feature_detector = None
        self.feature_extractor = None

        self.window_size = window_size  # parameter for bundle adjustment

        # input data
        self.n_views = n_views
        self.intrinsics = intrinsics
        self.correspondences = correspondences
        self.f2f_mapping = f2f_mapping
        self.images = images_dir

        # geometric verifcaiton
        self.hInliers = []
        self.eInliers = []
        self.fInliers = []

        # Estimation Results
        self.current_views = None  # the current views that are being processed, the key of dict: frame1_id_frame2_id
        self.extrinsics = None
        # data type dict: {frame1_id_frame2_id: ...} is useful to track frame to frame
        self.fundamental_matrix = dict()
        self.essential_matrix = dict()
        # frame-to-frame track the relative camera pose by setting the left view pose as [I|0] (assume the camera is at the origin in world coordinate)
        self.f2f_pose = dict()
        # the absolute camera pose in the world coordinate system, fix the first camera pose to be [I|0], stores in homogenous coordinates
        self.camera_pose = dict()

        # the dict is for 3d-2d feature track to solve pnp problem, the key is the triangulated 3d points from registered views, the values are the 2d points in the corresponding frame
        self.view_graph = ViewGraph()

    def feature_matching(self):
        # Feature Matching and Outlier rejection using RANSAC
        pass

    def estimate_fundamental_matrix(self, frame_to_frame: str):
        # Estimating Fundamental Matrix
        corrs = get_correspondences(self.correspondences, frame_to_frame)
        points1 = corrs[:, :2].numpy()
        points2 = corrs[:, 2:].numpy()
        F, mask = cv.findFundamentalMat(points1, points2, cv.RANSAC)
        if mask is not None:
            # remove outliers and update correspondences
            points1 = points1[mask.ravel() == 1]
            points2 = points2[mask.ravel() == 1]
        corrs = np.hstack((points1, points2))
        self.correspondences[frame_to_frame] = torch.tensor(corrs)
        return F, mask

    def estimate_essential_matrix(self, frame_to_frame: str):
        # Estimating Essential Matrix from Fundamental Matrix
        # E = K^(T) * F * K
        corrs = get_correspondences(self.correspondences, frame_to_frame)
        points1 = corrs[:, :2].numpy()
        points2 = corrs[:, 2:].numpy()
        E, mask = cv.findEssentialMat(
            points1, points2, self.intrinsics.numpy(), method=cv.RANSAC
        )
        if mask is not None:
            # remove outliers and update correspondences
            points1 = points1[mask.ravel() == 1]
            points2 = points2[mask.ravel() == 1]
        corrs = np.hstack((points1, points2))
        self.correspondences[frame_to_frame] = torch.tensor(corrs)

        return E, mask

    def estimate_camera_pose(self, frame_to_frame: str):
        # Decompose Camera Pose from Essential Matrix from a given frame to frame
        E = self.essential_matrix[frame_to_frame]
        R1, R2, t = cv.decomposeEssentialMat(E.numpy())
        # Consider 4 Possible Poses: R1T, R1(-T), R2T, R2(-T)
        return R1, R2, t

    def recover_pose(self, frame_to_frame) -> torch.tensor:
        # Check for Cheirality Condition using Triangulation
        # disambuiguate the 4 possible poses
        corrs = get_correspondences(self.correspondences, frame_to_frame)
        points1 = corrs[:, :2].numpy()
        points2 = corrs[:, 2:].numpy()
        E = self.essential_matrix[frame_to_frame]
        E = E.numpy().astype(np.float64)
        K = self.intrinsics.numpy().astype(np.float64)
        _, R, t, _ = cv.recoverPose(
            E,
            points1.astype(np.float64),
            points2.astype(np.float64),
            cameraMatrix=K,
        )
        M_ext = torch.cat((torch.tensor(R), torch.tensor(t)), dim=1)
        M_ext = torch.cat((M_ext, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)))

        return M_ext

    # triangulate 3D points from 2D correspondences from given project matrices
    # M_ext1, M_ext2: 3x4 extrinsic matrices in the euclidean coordinate
    def triangulate(self, points1: torch.tensor, points2: torch.tensor, M_ext1, M_ext2):
        if M_ext1.shape[0] == 4:
            M_ext1 = M_ext1[:-1, :]
        if M_ext2.shape[0] == 4:
            M_ext2 = M_ext2[:-1, :]

        # Triangulate the 3D points
        projection1 = self.intrinsics @ M_ext1.to(torch.float32)
        projection2 = self.intrinsics @ M_ext2.to(torch.float32)

        points1 = points1.numpy().astype(np.float32)
        points2 = points2.numpy().astype(np.float32)

        # homogeneous coordinates
        points3D = (
            cv.triangulatePoints(
                projection1.numpy(),
                projection2.numpy(),
                points1.T.copy(),
                points2.T.copy(),
            ).T
        ).astype(np.float32)

        # normalized the homogeneous coordinates
        normalized_points3D = points3D / points3D[:, -1, None]
        # remove the outliers with negative depth
        normalized_points3D = normalized_points3D[normalized_points3D[:, -1] > 0]

        return normalized_points3D

    # def triangulate(self, points1: torch.tensor, points2: torch.tensor, M_ext1, M_ext2):
    #     if M_ext1.shape[0] == 4:
    #         M_ext1 = M_ext1[:-1, :]
    #     if M_ext2.shape[0] == 4:
    #         M_ext2 = M_ext2[:-1, :]

    #     # Triangulate the 3D points
    #     points1 = points1.numpy().astype(np.float32)
    #     points2 = points2.numpy().astype(np.float32)
    #     points1 = np.concatenate((points1, np.ones((points1.shape[0], 1))), axis=-1)
    #     points2 = np.concatenate((points2, np.ones((points2.shape[0], 1))), axis=-1)
    #     norm_point1, norm_point2 = normalize_pts(
    #         self.intrinsics, self.intrinsics, points1, points2
    #     )
    #     points3D = linear_triangulation(
    #         norm_point1, norm_point2, M_ext1.numpy(), M_ext2.numpy()
    #     )
    #     homo_points3D = np.concatenate(
    #         (points3D, np.ones((points3D.shape[0], 1))), axis=-1
    #     )
    #     return homo_points3D

    def estimate_homography(self, frame_to_frame: str):
        # Estimating Homography Matrix
        corrs = get_correspondences(self.correspondences, frame_to_frame)
        points1 = corrs[:, :2].numpy()
        points2 = corrs[:, 2:].numpy()
        H, mask = cv.findHomography(points1, points2, cv.RANSAC)
        return H, mask

    # valid geometry verified image pairs: N_e / N_f > epsilon_ef and  N_h / n_E < epsilon_he
    # select geometriy valid pairs and return the highest score_he as the initial pair
    def geometry_verification(self, epsilon_ef=None, epsilon_he=None) -> str:
        # TODO: unknown threshold epsilon_ef and epsilon_he
        for frame_to_frame in self.f2f_mapping:
            _, mask = self.estimate_homography(frame_to_frame)
            self.hInliers.append(np.sum(mask, dtype=np.int32))

        score_ef = torch.tensor(self.eInliers) / torch.tensor(self.fInliers)
        score_he = torch.tensor(self.hInliers) / torch.tensor(self.eInliers)
        if epsilon_ef is not None and epsilon_he is not None:
            # TODO: select valid pairs
            valid_mask = (score_ef > epsilon_ef) and (score_he < epsilon_he)

        # sort the scores and select the best pair as the initial pair
        _, f2f_index = torch.sort(score_he)
        initial_view_pair = list(self.correspondences.keys())[f2f_index[0]]
        return initial_view_pair

    # select next view and return the 2d-3d correspondences for pnp
    def select_next_view(
        self, H=1080, W=1920
    ) -> tuple[str, str, torch.tensor, torch.tensor]:
        # pick the one can has most 3D points
        visible_points = []
        potential_next_views = []

        for registered_view in self._get_registered_cameras():
            potential_next_views.extend(
                [
                    f2f_key
                    for f2f_key in self.f2f_mapping
                    if registered_view in get_frames_from_key(f2f_key)
                    and (
                        get_frames_from_key(f2f_key)[0]
                        not in self._get_registered_cameras()
                        or get_frames_from_key(f2f_key)[1]
                        not in self._get_registered_cameras()
                    )
                ]
            )

        # NOTE: project the triangulated 3D point to the camera view and check the visibility
        for view in potential_next_views:
            self.estimate_essential_matrix(view)
            M_ext1 = torch.eye(4, dtype=torch.float32)
            M_ext2 = self.recover_pose(view)
            R = cv.Rodrigues(M_ext2[:-1, :3].numpy())
            t = M_ext2[:-1, 3].numpy()
            points_3d = np.array(self._get_points3D(), dtype=np.float32)[:, :-1]
            # a weird bug if the numpy array is not copied
            points = np.copy(points_3d)
            projectedPoints, _ = cv.projectPoints(
                points,
                R[0],
                t,
                K.numpy().astype(np.float64),
                None,
            )
            # check visibility of 2D points
            projectedPoints = projectedPoints.squeeze()
            visibility_mask = (
                (projectedPoints[:, 0] > 0)
                & (projectedPoints[:, 0] < W)
                & (projectedPoints[:, 1] > 0)
                & (projectedPoints[:, 1] < H)
            )
            projectedPoints = projectedPoints[visibility_mask]
            visible_points.append(len(projectedPoints))

        # select the unregistered view with the most visible points as the next view
        sorted_index = np.argsort(visible_points)[::-1]
        next_view_pair = potential_next_views[sorted_index[0]]
        view_pair = get_frames_from_key(next_view_pair)
        next_view = [i for i in view_pair if i not in self._get_registered_cameras()][0]

        return next_view

    """
        find the 2d correspondences from registered views with the next view
        return the 2d correspondences and its 3d correspondences
    """

    def track_3d_2d_correspondences(self, next_view, diff_threshold=0):
        node_3d_ids = []  # list of 3d node id
        node_2d_ids = []  # list of 2d node id

        corres_2d = []  # list of 2d correspondences (position, frame_id)

        pos_corr_3d = []  # list for 3d position
        pos_corr_2d = []  # list for 2d position

        # find all exists view match with the next view
        match_pairs = []
        for f2f in self.f2f_mapping:
            k1, k2 = get_frames_from_key(f2f)
            if next_view in (k1, k2):
                match_pairs.append(f2f)

        # iterate the 2d correspondences from registered views in view graph to find the same keypoint in next view
        for subnode in self.view_graph.sub_nodes.values():
            registered_frame = subnode.frame_id
            if registered_frame == next_view:
                continue
            if int(registered_frame) > int(next_view):
                f2f_key = f"{next_view}_{registered_frame}"
            else:
                f2f_key = f"{registered_frame}_{next_view}"
            # check if there is match pair between subnode.frame and the next view
            if f2f_key not in match_pairs:
                continue

            if f2f_key in match_pairs:
                corresondences_2d = get_correspondences(self.correspondences, f2f_key)
                key1, key2 = get_frames_from_key(f2f_key)
                if key1 == registered_frame:
                    registered_cols = 0
                    unregistered_cols = 2
                else:
                    registered_cols = 2
                    unregistered_cols = 0
                # NOTE: for stage1, the 2d correspondences has the exact pixel position across views. In stage2, we may consider the threshold
                idx = torch.nonzero(
                    (
                        (
                            torch.abs(
                                corresondences_2d[
                                    :, registered_cols : registered_cols + 2
                                ]
                                - torch.tensor(subnode.position)
                            )
                            <= diff_threshold
                        ).all(dim=1)
                    )
                )

                # find matched 2d correspondences from this registered view
                if len(idx) > 0:
                    pos = corresondences_2d[
                        idx[0], unregistered_cols : unregistered_cols + 2
                    ]
                    node_3d_ids.append(subnode.parent_node)

                    pos_corr_3d.append(
                        torch.tensor(
                            self.view_graph.nodes[subnode.parent_node].position
                        )
                    )

                    pos_corr_2d.extend(torch.tensor(pos))
                    corres_2d.append((pos, next_view))
                    node_2d_ids.append(subnode.id)

        self._update_feature_tracK(node_3d_ids, next_view, pos_corr_2d)

        if len(node_2d_ids) == 0:
            return None
        else:
            return (
                node_2d_ids,
                torch.stack(pos_corr_2d),
                node_3d_ids,
                torch.stack(pos_corr_3d),
            )

    def solvePnP(
        self,
        frame_id: str,
        subnode_ids: list,
        corres_2d: torch.tensor,
        node_ids: list,
        corres_3d: torch.tensor,
    ):
        # Perspective-n-Point: the 6 DOF camera pose can be estimated using linear least squares with RANSAC for outlier rejection
        points3D = corres_3d[:, :3]
        points2D = corres_2d.numpy()
        points2D = np.copy(points2D)
        points3D = np.copy(points3D)

        _, rotation_vector, translation_vector, inliers = cv.solvePnPRansac(
            points3D.astype(np.float32),
            points2D.astype(np.float32),
            self.intrinsics.numpy().astype(np.float32),
            None,
        )
        R, _ = cv.Rodrigues(rotation_vector)  # convert to rotation matrix
        M_ext = torch.cat((torch.tensor(R), torch.tensor(translation_vector)), dim=1)
        M_ext = torch.cat((M_ext, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)))
        self.camera_pose[frame_id] = M_ext
        print(f"registered NO.{len(self._get_registered_cameras())} view: {frame_id}")
        # TODO: remove the outliers in the subset of 3d points and 2d points
        if inliers is not None:
            outliers = torch.nonzero(
                ((torch.arange(len(corres_3d))[:, None] - inliers.squeeze()) != 0).all(
                    dim=1
                )
            )

            if len(outliers) > 0:
                for outlier in outliers:
                    # TODO: remove 2d correspondences from the matching pairs
                    # remove 3d-2d correspondences from the view graph
                    self.view_graph.remove_node(node_ids[outlier])

        return M_ext

    def bundle_adjustment(self, window_size=5):
        # Refine poses from last n registered cameras and 3D points together, initialized by previous reconstruction by minimizing reporjection error.
        K = self.intrinsics.numpy().astype(np.float32)

        last_n_registered_cams = self._get_registered_cameras()[-window_size:]
        # m*12 paramteres for camera poses
        param_poses = torch.stack(
            [self.camera_pose[view].flatten()[:12] for view in last_n_registered_cams]
        )
        camera_node_pairs = (
            dict()
        )  # dict of camera node pairs [frame_id:(node.position, subnode.position), ...]

        # n*4 parameters for 3D points
        param_points3d = []
        num_points_3d_per_camera = []
        point_3d_ids = []
        for view in last_n_registered_cams:
            node_pairs = self.view_graph.search_nodes_by_frame(
                view
            )  # list of (node_id, subnode_id)
            camera_node_pairs[view] = np.vstack(
                [
                    self.view_graph.sub_nodes[subnode_id].position.numpy().squeeze()
                    for _, subnode_id in node_pairs
                ]
            )
            param_points3d.extend(
                [self.view_graph.nodes[node_id].position for node_id, _ in node_pairs]
            )
            point_3d_ids.extend([node_id for node_id, _ in node_pairs])
            num_points_3d_per_camera.append(len(node_pairs))

        num_camera, num_points_3d = window_size, len(param_points3d)

        # (window_size*12 + n*4) parameters
        params = np.concatenate(
            (param_poses.flatten(), np.array(param_points3d).flatten())
        )

        # optimized param is the flattened array of n camera extrinsics and m 3D points
        # reprojection error for all cameras and triangulated 3d point: sum_isum_j () Vij(||P_i(X_j)-x_ij||))
        def cost(
            params,
            num_camera,
            num_points_3d,
            K,
            registered_cams,
            camera_node_pairs,
            num_points_3d_per_camera,
        ):
            error = []
            camera_poses = params[: num_camera * 12].reshape(num_camera, 3, 4)
            points_3d = params[num_camera * 12 :].reshape(num_points_3d, 4)

            projections = np.matmul(K, camera_poses)

            for idx, total in enumerate(num_points_3d_per_camera):
                M_projection = projections[idx]
                points_2d = camera_node_pairs[registered_cams[idx]]
                scene_points = points_3d[
                    sum(num_points_3d_per_camera[:idx]) : sum(
                        num_points_3d_per_camera[:idx]
                    )
                    + total,
                    :,
                ]
                projected_points = np.matmul(M_projection, scene_points.T).T
                normalized_projected_points = (
                    projected_points / projected_points[:, 2, None]
                )
                normalized_projected_points = normalized_projected_points[:, :2].astype(
                    np.float32
                )

                reprojection_error = np.linalg.norm(
                    normalized_projected_points - points_2d, axis=1
                )

                error.append(reprojection_error.mean())

            return np.array(error).mean()

        res = least_squares(
            cost,
            params,
            verbose=2,
            x_scale="jac",
            xtol=1e-4,
            ftol=1e-4,
            method="trf",
            args=(
                window_size,
                num_points_3d,
                K,
                last_n_registered_cams,
                camera_node_pairs,
                num_points_3d_per_camera,
            ),
        )

        M_ext = torch.tensor(res.x[: num_camera * 12].reshape(num_camera, 3, 4))
        # reshape to homonegenous coordinates
        M_ext = torch.cat(
            (M_ext, torch.tensor([[[0, 0, 0, 1]]] * num_camera, dtype=torch.float32)),
            dim=1,
        )
        points3D = res.x[num_camera * 12 :].reshape(num_points_3d, 4)

        # update camera pose and 3D points
        for i, view in enumerate(last_n_registered_cams):
            self.camera_pose[view] = M_ext[i]

        for node_id, node_position in zip(point_3d_ids, points3D):
            self.view_graph.nodes[node_id].position = node_position

    def run(self, data, test_frame):
        # Run the entire pipeline
        if self.correspondences is None:
            self.feature_matching()
        for frame_to_frame in self.f2f_mapping:
            F, f_mask = self.estimate_fundamental_matrix(frame_to_frame)
            E, e_mask = self.estimate_essential_matrix(frame_to_frame)
            if F is None or E is None:
                self.f2f_mapping.remove(frame_to_frame)
                self.correspondences.pop(frame_to_frame)
                continue
            self.essential_matrix[frame_to_frame] = torch.tensor(E, dtype=torch.float32)
            self.fundamental_matrix[frame_to_frame] = torch.tensor(
                F, dtype=torch.float32
            )

            self.fInliers.append(np.sum(f_mask, dtype=np.int32))
            self.eInliers.append(np.sum(e_mask, dtype=np.int32))
        initial_pair_views = self.geometry_verification()
        self.current_views = initial_pair_views
        first_view, second_view = get_frames_from_key(initial_pair_views)
        M_ext1 = torch.eye(4, dtype=torch.float32)
        M_ext2 = self.recover_pose(initial_pair_views)
        self.camera_pose[first_view] = M_ext1
        self.camera_pose[second_view] = M_ext2
        # triangulate the 3D points from the initial pair of views
        matches = get_correspondences(self.correspondences, initial_pair_views)
        points3D = self.triangulate(matches[:, :2], matches[:, 2:], M_ext1, M_ext2)
        # add the 3D-2D correspondences to feature track dict
        self._add_feature_track(
            points3D, first_view, second_view, matches[:, :2], matches[:, 2:]
        )
        self.view_graph.save_to_json("vg_2.json")
        save_camera_poses(self.camera_pose, f"{data}_test2.json")
        save_point_to_ply(
            np.array(self._get_points3D()),
            f"{data}_test2.ply",
        )
        self.bundle_adjustment(window_size=2)
        self.view_graph.save_to_json("vg_2_opt.json")
        save_camera_poses(self.camera_pose, f"{data}_test2_opt.json")
        save_point_to_ply(
            np.array(self._get_points3D()),
            f"{data}_test2_opt.ply",
        )

        for count_view in range(1, test_frame - 1):
            next_view = self.select_next_view()
            # track the 3d-2d correspondences in the view graph
            track_corr = self.track_3d_2d_correspondences(next_view)

            if track_corr is None or track_corr[1].shape[0] < 4:
                # decompose essential matrix to get camera pose
                for registered_view in self._get_registered_cameras():
                    if int(registered_view) < int(next_view):
                        next_frame_pair = f"{registered_view}_{next_view}"
                    else:
                        next_frame_pair = f"{next_view}_{registered_view}"
                    if next_frame_pair in self.f2f_mapping:
                        M_ext = self.recover_pose(next_frame_pair)
                        self.camera_pose[next_view] = M_ext
                        print(
                            f"registered NO.{len(self._get_registered_cameras())} view: {next_view}"
                        )
            else:
                subnode_ids, corres_2d, node_ids, corres_3d = track_corr
                self.solvePnP(next_view, subnode_ids, corres_2d, node_ids, corres_3d)
            # find the correspondences between the registered views and the next view
            triangule_f2f = []
            for registered_view in self._get_registered_cameras()[:-1]:
                if int(registered_view) < int(next_view):
                    next_frame_pair = f"{registered_view}_{next_view}"
                else:
                    next_frame_pair = f"{next_view}_{registered_view}"
                if next_frame_pair in self.f2f_mapping:
                    triangule_f2f.append(next_frame_pair)

            for next_frame_pair in triangule_f2f:
                unregisterd_idx = get_frames_from_key(next_frame_pair).index(next_view)
                registered_idx = 1 - unregisterd_idx
                frame1, frame2 = (
                    get_frames_from_key(next_frame_pair)[registered_idx],
                    next_view,
                )
                M_ext1 = self.camera_pose[frame1]
                M_ext2 = self.camera_pose[frame2]
                matches = get_correspondences(self.correspondences, next_frame_pair)
                point1 = matches[:, registered_idx * 2 : registered_idx * 2 + 2]

                nonexist_mask = []
                # list of 3d node id
                exist_node_ids = []
                for point in point1:
                    exist = self.view_graph._subnode_exists(point, frame1)
                    if exist is not None:
                        exist_node_ids.append(exist.parent_node)
                    nonexist_mask.append(not exist)

                # triangulate the new scene point
                if np.sum(nonexist_mask) > 0:
                    new_corres_pair = matches[nonexist_mask]
                    points3D = self.triangulate(
                        new_corres_pair[:, :2], new_corres_pair[:, 2:], M_ext1, M_ext2
                    )
                    # update the feature track dict
                    self._add_feature_track(
                        points3D,
                        frame1,
                        frame2,
                        new_corres_pair[:, :2],
                        new_corres_pair[:, 2:],
                    )

            # bundle adjust for every n frames
            if (count_view + 2) % self.window_size == 0:
                self.bundle_adjustment(window_size=self.window_size)

        self.view_graph.save_to_json(f"vg_{test_frame}_opt.json")
        save_camera_poses(self.camera_pose, f"{data}_test{test_frame}_opt.json")
        save_point_to_ply(
            np.array(self._get_points3D()),
            f"{data}_test{test_frame}_opt.ply",
        )

    """ 
        track the 3d-2d correspondences in the view graph
        if the 3d points are not triangulated before, we need to create a new node in the view graph and add the 2d correspondences
        else we just need to add the 2d correspondences to the feature track dict
    """

    def seqential_run(self, data="box", test_frame=5):
        if test_frame is None:
            test_frame = self.n_views
        img_root = f"data/{data}/images"
        img_files = os.listdir(img_root)
        sequential = sorted([int(i[:5]) for i in img_files])
        sequential = [str(i) for i in sequential]

        # Run the entire pipeline
        for frame_to_frame in self.f2f_mapping:
            F, f_mask = self.estimate_fundamental_matrix(frame_to_frame)
            E, e_mask = self.estimate_essential_matrix(frame_to_frame)
            if F is None or E is None:
                self.f2f_mapping.remove(frame_to_frame)
                del self.correspondences[frame_to_frame]
                continue
            self.essential_matrix[frame_to_frame] = torch.tensor(E, dtype=torch.float32)
            self.fundamental_matrix[frame_to_frame] = torch.tensor(
                F, dtype=torch.float32
            )

            self.fInliers.append(np.sum(f_mask, dtype=np.int32))
            self.eInliers.append(np.sum(e_mask, dtype=np.int32))
        initial_pair_views = f"{sequential[:2][0]}_{sequential[:2][1]}"
        self.current_views = initial_pair_views
        first_view, second_view = get_frames_from_key(initial_pair_views)
        M_ext1 = torch.eye(4, dtype=torch.float32)
        M_ext2 = self.recover_pose(initial_pair_views)
        self.camera_pose[first_view] = M_ext1
        self.camera_pose[second_view] = M_ext2
        # triangulate the 3D points from the initial pair of views
        matches = get_correspondences(self.correspondences, initial_pair_views)
        points3D = self.triangulate(matches[:, :2], matches[:, 2:], M_ext1, M_ext2)
        # add the 3D-2D correspondences to feature track dict
        self._add_feature_track(
            points3D, first_view, second_view, matches[:, :2], matches[:, 2:]
        )
        # self.view_graph.save_to_json(f"{data}_vg_2.json")
        # save_camera_poses(self.camera_pose, f"{data}_test2.json")
        # save_point_to_ply(
        #     np.array(self._get_points3D()),
        #     f"{data}_test2.ply",
        # )
        self.bundle_adjustment(window_size=2)

        for count_view in range(1, test_frame - 1):
            next_view = sequential[count_view + 1]
            # track the 3d-2d correspondences in the view graph
            track_corr = self.track_3d_2d_correspondences(next_view)
            if track_corr is None:
                # decompose essential matrix to get camera pose
                previous_view = sequential[count_view]
                if int(previous_view) < int(next_view):
                    M_ext = self.recover_pose(f"{previous_view}_{next_view}")
                else:
                    M_ext = self.recover_pose(f"{next_view}_{previous_view}")
                self.camera_pose[next_view] = self.camera_pose[previous_view] @ M_ext
            else:
                subnode_ids, corres_2d, node_ids, corres_3d = track_corr
                self.solvePnP(next_view, subnode_ids, corres_2d, node_ids, corres_3d)
            # find the correspondences between the registered views and the next view
            triangule_f2f = []
            for registered_view in self._get_registered_cameras()[:-1]:
                if int(registered_view) < int(next_view):
                    next_frame_pair = f"{registered_view}_{next_view}"
                else:
                    next_frame_pair = f"{next_view}_{registered_view}"
                if next_frame_pair in self.f2f_mapping:
                    triangule_f2f.append(next_frame_pair)

            for next_frame_pair in triangule_f2f:
                unregisterd_idx = get_frames_from_key(next_frame_pair).index(next_view)
                registered_idx = 1 - unregisterd_idx
                frame1, frame2 = (
                    get_frames_from_key(next_frame_pair)[registered_idx],
                    next_view,
                )
                M_ext1 = self.camera_pose[frame1]
                M_ext2 = self.camera_pose[frame2]
                matches = get_correspondences(self.correspondences, next_frame_pair)
                point1 = matches[:, registered_idx * 2 : registered_idx * 2 + 2]

                nonexist_mask = []
                # list of 3d node id
                exist_node_ids = []
                for point in point1:
                    exist = self.view_graph._subnode_exists(point, frame1)
                    if exist is not None:
                        exist_node_ids.append(exist.parent_node)
                    nonexist_mask.append(not exist)

                # triangulate the new scene point
                new_corres_pair = matches[nonexist_mask]
                points3D = self.triangulate(
                    new_corres_pair[:, :2], new_corres_pair[:, 2:], M_ext1, M_ext2
                )
                # update the feature track dict
                self._add_feature_track(
                    points3D,
                    frame1,
                    frame2,
                    new_corres_pair[:, :2],
                    new_corres_pair[:, 2:],
                )

            # bundle adjust for every n frames
            if (count_view + 2) % self.window_size == 0:
                self.bundle_adjustment(window_size=self.window_size)

        self.view_graph.save_to_json(f"vg_{test_frame}_opt.json")
        save_camera_poses(self.camera_pose, f"{data}_test{test_frame}_opt.json")
        save_point_to_ply(
            np.array(self._get_points3D()),
            f"{data}_test{test_frame}_opt.ply",
        )

    def seqential_corr_run(self, seq, data="box", test_frame=46):
        sequential = seq

        # Run the entire pipeline
        for frame_to_frame in self.f2f_mapping:
            F, f_mask = self.estimate_fundamental_matrix(frame_to_frame)
            E, e_mask = self.estimate_essential_matrix(frame_to_frame)
            self.essential_matrix[frame_to_frame] = torch.tensor(E, dtype=torch.float32)
            self.fundamental_matrix[frame_to_frame] = torch.tensor(
                F, dtype=torch.float32
            )

            self.fInliers.append(np.sum(f_mask, dtype=np.int32))
            self.eInliers.append(np.sum(e_mask, dtype=np.int32))
        initial_pair_views = f"{sequential[:2][0]}_{sequential[:2][1]}"
        self.current_views = initial_pair_views
        first_view, second_view = get_frames_from_key(initial_pair_views)
        M_ext1 = torch.eye(4, dtype=torch.float32)
        M_ext2 = self.recover_pose(initial_pair_views)
        self.camera_pose[first_view] = M_ext1
        self.camera_pose[second_view] = M_ext2
        # triangulate the 3D points from the initial pair of views
        matches = get_correspondences(self.correspondences, initial_pair_views)
        points3D = self.triangulate(matches[:, :2], matches[:, 2:], M_ext1, M_ext2)
        # add the 3D-2D correspondences to feature track dict
        self._add_feature_track(
            points3D, first_view, second_view, matches[:, :2], matches[:, 2:]
        )
        self.bundle_adjustment(window_size=2)

        for count_view in range(1, test_frame - 1):
            next_view = sequential[count_view + 1]
            # track the 3d-2d correspondences in the view graph
            for sub in self.view_graph.sub_nodes.values():
                if sub.parent_node == None:
                    self.view_graph.save_to_json("vg_buggy.json")

            track_corr = self.track_3d_2d_correspondences(next_view)
            if track_corr is None:
                # decompose essential matrix to get camera pose
                previous_view = sequential[count_view]
                if int(previous_view) < int(next_view):
                    M_ext = self.recover_pose(f"{previous_view}_{next_view}")
                else:
                    M_ext = self.recover_pose(f"{next_view}_{previous_view}")
                self.camera_pose[next_view] = self.camera_pose[previous_view] @ M_ext
            else:
                subnode_ids, corres_2d, node_ids, corres_3d = track_corr
                self.solvePnP(next_view, subnode_ids, corres_2d, node_ids, corres_3d)
            # find the correspondences between the registered views and the next view
            triangule_f2f = []
            for registered_view in self._get_registered_cameras()[:-1]:
                if int(registered_view) < int(next_view):
                    next_frame_pair = f"{registered_view}_{next_view}"
                else:
                    next_frame_pair = f"{next_view}_{registered_view}"
                if next_frame_pair in self.f2f_mapping:
                    triangule_f2f.append(next_frame_pair)

            for next_frame_pair in triangule_f2f:
                unregisterd_idx = get_frames_from_key(next_frame_pair).index(next_view)
                registered_idx = 1 - unregisterd_idx
                frame1, frame2 = (
                    get_frames_from_key(next_frame_pair)[registered_idx],
                    next_view,
                )
                M_ext1 = self.camera_pose[frame1]
                M_ext2 = self.camera_pose[frame2]
                matches = get_correspondences(self.correspondences, next_frame_pair)
                point1 = matches[:, registered_idx * 2 : registered_idx * 2 + 2]

                nonexist_mask = []
                # list of 3d node id
                exist_node_ids = []
                for point in point1:
                    exist = self.view_graph._subnode_exists(point, frame1)
                    if exist is not None:
                        exist_node_ids.append(exist.parent_node)
                    nonexist_mask.append(not exist)

                # add edge to the view graph for the exsiting 3D points
                if len(exist_node_ids) > 0:
                    # update the feature track dict
                    exist_mask = [not m for m in nonexist_mask]
                    self._update_feature_tracK(
                        exist_node_ids,
                        next_view,
                        matches[exist_mask][
                            :, unregisterd_idx * 2 : unregisterd_idx * 2 + 2
                        ],
                    )

                # triangulate the new scene point
                new_corres_pair = matches[nonexist_mask]
                points3D = self.triangulate(
                    new_corres_pair[:, :2], new_corres_pair[:, 2:], M_ext1, M_ext2
                )
                for sub in self.view_graph.sub_nodes.values():
                    assert sub.parent_node != None
                # update the feature track dict
                self._add_feature_track(
                    points3D,
                    frame1,
                    frame2,
                    new_corres_pair[:, :2],
                    new_corres_pair[:, 2:],
                )
                for sub in self.view_graph.sub_nodes.values():
                    if sub.parent_node == None:
                        self.view_graph.save_to_json("vg_buggy.json")

            # bundle adjust for every 5 frames
            window_size = 4
            if (count_view + 2) % window_size == 0:
                self.bundle_adjustment(window_size=window_size)

    # add nodes and edge to the view graph for the new triangulated 3D points
    def _add_feature_track(
        self, points_3d, frame1, frame2, points_2d1, points_2d2
    ) -> None:
        corres_3d = self.view_graph.create_multi_nodes(points_3d)
        corres_2d_frame1 = [
            self.view_graph.create_subnode(p, frame1) for p in points_2d1
        ]
        corres_2d_frame2 = [
            self.view_graph.create_subnode(p, frame2) for p in points_2d2
        ]

        self.view_graph.add_multi_edges(
            [node.id for node in corres_3d], [sub.id for sub in corres_2d_frame1]
        )

        self.view_graph.add_multi_edges(
            [node.id for node in corres_3d], [sub.id for sub in corres_2d_frame2]
        )

    # add edge to the view graph for the exsiting 3D points
    def _update_feature_tracK(self, node_id, frame, points_2d):
        for sub in self.view_graph.sub_nodes.values():
            if sub.parent_node == None:
                self.view_graph.save_to_json("vg_buggy.json")
        subnodes = [self.view_graph.create_subnode(p, frame) for p in points_2d]
        self.view_graph.add_multi_edges(node_id, [sub.id for sub in subnodes])
        for sub in self.view_graph.sub_nodes.values():
            if sub.parent_node == None:
                self.view_graph.save_to_json("vg_buggy.json")

    def _get_registered_cameras(self):
        return list(self.camera_pose.keys())

    """
        get the triangulated 3D points from the view graph
    """

    def _get_points3D(self) -> list:
        return [node.position for node in self.view_graph.nodes.values()]

    def _generate_most_corr_seq(self) -> list:
        # define seqence in the order with max correspondences
        estimation_seq = []
        estimation_f2fs = []

        len_corr = torch.tensor([len(corr) for corr in corr_dicts.values()])
        first_view_pair = len_corr.argmax()
        # define the next sequence
        frame1, frame2 = get_frames_from_key(f2f[first_view_pair], "str")
        estimation_seq.append(frame1)
        estimation_seq.append(frame2)
        estimation_f2fs.append(f2f[first_view_pair])

        for _ in range(num_views - 2):
            candidate_pairs_len = []
            candidate_pair = []
            for registered_view in estimation_seq:
                registered_pair = [
                    pair
                    for pair in f2f
                    if (
                        int(registered_view) in get_frames_from_key(pair, "int")
                        and (
                            get_frames_from_key(pair, "str")[0] not in estimation_seq
                            or get_frames_from_key(pair, "str")[1] not in estimation_seq
                        )
                    )
                ]

                candidate_pairs_len.extend(
                    [len(corr_dicts[pair]) for pair in registered_pair]
                )
                candidate_pair.extend(registered_pair)

            idx = torch.tensor(candidate_pairs_len).argmax()
            estimation_f2fs.append(candidate_pair[idx])
            next_view = [
                i
                for i in get_frames_from_key(candidate_pair[idx], "str")
                if i not in estimation_seq
            ][0]

        estimation_seq.append(next_view)

        return estimation_seq


if __name__ == "__main__":
    data = "milk"
    corr_dicts, f2f = read_correspondences(f"./data/{data}/correspondences")

    K = get_intrinsics(f"./data/{data}/gt_camera_parameters.json")
    num_views = len(os.listdir(f"./data/{data}/images"))
    # K = get_intrinsics(f"./data/{data}/camera_parameters.json")
    sfm = SFM(
        correspondences=corr_dicts,
        intrinsics=K,
        f2f_mapping=f2f,
        n_views=num_views,
        images_dir=f"./data/{data}/images",
    )

    # sfm.seqential_corr_run(seq=estimation_seq, data=data)
    sfm.seqential_run(data,test_frame=num_views)
    # sfm.run(data, test_frame=10)
