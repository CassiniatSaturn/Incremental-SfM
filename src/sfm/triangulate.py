import numpy as np


def homo_pts(pts: np.array):
    return pts, np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)


def normalize_pts(K_A, K_B, pts1: np.array, pts2: np.array):

    # find translation and scaling for two points
    pts1_norm = (np.linalg.inv(K_A) @ pts1.T).T
    pts2_norm = (np.linalg.inv(K_B) @ pts2.T).T
    # plt.scatter(pts1_norm[:, 0], pts1_norm[:, 1], c='g', marker='o', s=5)
    # plt.scatter(pts2_norm[:, 0], pts2_norm[:, 1], c='r', marker='o', s=5)
    # plt.show()
    return pts1_norm, pts2_norm


def linear_triangulation(pts1: np.array, pts2: np.array, P: np.matrix, pose: np.matrix):
    # P = np.eye(3,4)
    P_dash = pose
    points_3D = []
    num_points = pts1.shape[0]
    for i in range(num_points):
        # print(pts1.shape)
        point1 = pts1[i]
        point2 = pts2[i]
        # print(point1.shape, point2.shape)
        """
        A = np.array([
            (point1[Y] * P[ROW3]) - P[ROW2],
            P[ROW1] - (point1[X]*P[ROW3]),
            (point2[Y] * P_dash[ROW3]) - P_dash[ROW2],
            P_dash[ROW1] - (point2[X] * P_dash[ROW3])
        ])
        """
        point1_cross = np.array(
            [
                [0, -point1[2], point1[1]],
                [point1[2], 0, -point1[0]],
                [-point1[1], point1[0], 0],
            ]
        )

        point2_cross = np.array(
            [
                [0, -point2[2], point2[1]],
                [point2[2], 0, -point2[0]],
                [-point2[1], point2[0], 0],
            ]
        )

        point1_cross_P = point1_cross @ P
        point2_cross_P_dash = point2_cross @ P_dash

        A = np.vstack((point1_cross_P, point2_cross_P_dash))

        _, _, VT = np.linalg.svd(A)
        solution = VT.T[:, -1]
        solution /= solution[-1]

        points_3D.append([solution[0], solution[1], solution[2]])

    # print(points_3D)
    return np.array(points_3D)


# corres_A, corres_B = homo_pts(corres_A)[1], homo_pts(corres_B)[1]
# corres_A, corres_B = normalize_pts(K_A, K_B, corres_A, corres_B)
# points = linear_triangulation(corres_A, corres_B, w2c_A[:3, :4], w2c_B[:3, :4])
