import numpy as np
import os
import numpy as np
import numpy as np

def normalize_keypoints(kps):
    """
    Normalize the keypoints
    Input: 
    pts: the keypoints
    Output:
    pts_norm: the normalized keypoints
    T: the transformation matrix
    """
    num_pts = kps.shape[0]
    kps_h = np.hstack((kps, np.ones((num_pts, 1))))

    centroid = np.mean(kps, axis=0)
    mean_dist = np.sqrt(np.mean(np.sum((kps - centroid)**2, axis=1), axis=0)) #np.mean(np.sqrt(np.sum((kps - centroid) ** 2, axis=1)))

    scale = np.sqrt(2) / mean_dist
    Tran1 = np.diagflat(np.array([scale, scale, 1]))
    Tran2 = np.column_stack((np.row_stack((np.eye(2), [[0, 0]])), [-centroid[0], -centroid[1], 1]))
    T = Tran1 @ Tran2
    kps_norm = (T @ kps_h.T) # (3,N)

    return kps_norm, T

def findFundamentalM(kps1, kps2):
    """
    Estimate the fundamental matrix from left-right image point correspondences using 8-point algorithm
    Outputs denormalized and scaled fundamental matrix
    Input: 
    kps1: keypoints of the left image (N,2)
    kps2: keypoints of the right image (N,2)
    Output:
    F: the estimated fundamental matrix (3,3)
    """
    num_matches = kps1.shape[0]
    kps1_norm, T1 = normalize_keypoints(kps1)
    kps2_norm, T2 = normalize_keypoints(kps2)

    A = np.zeros((num_matches, 9))

    for i in range(num_matches):
        x1 = np.array([kps1_norm[:, i]]).reshape((3, 1))
        x2 = np.array([kps2_norm[:, i]]).reshape((3, 1))
        row = (x2 @ x1.T).reshape(1, 9)
        A[i] = row

    # kps1_norm = kps1_norm.T
    # kps2_norm = kps2_norm.T
    # B = np.array([
    #         kps1_norm[0, :] * kps2_norm[0, :], kps1_norm[0, :] * kps2_norm[1, :], kps1_norm[0, :] * kps2_norm[2, :],
    #         kps1_norm[1, :] * kps2_norm[0, :], kps1_norm[1, :] * kps2_norm[1, :], kps1_norm[1, :] * kps2_norm[2, :],
    #         kps1_norm[2, :] * kps2_norm[0, :], kps1_norm[2, :] * kps2_norm[1, :], kps1_norm[2, :] * kps2_norm[2, :]
    #     ]).T

    U, sings, Vh = np.linalg.svd(A)
    V = Vh.T
    F_estimated = V[:, -1].reshape(3,3)
    
    # recalculate estimated F after enforcing the rank 2 constraint
    U, sings, Vh = np.linalg.svd(F_estimated)
    sings[-1] = 0
    F_est = U @ np.diag(sings) @ Vh

    # denormalize F_est since we used normalized keypoints
    F_est = T2.T @ F_est @ T1

    # scale the fundamental matrix since the last element might not be 1
    F_est = F_est * (1 / F_est[2, 2]) 

    return F_est

def findFundamentalM_RANSAC(kps1, kps2, adaptive=False):
    """
    Estimate the fundamental matrix from left-right image point correspondences using RANSAC
    Input: 
    kps1: keypoints of the left image (N,2)
    kps2: keypoints of the right image (N,2)
    Output:
    F: the estimated fundamental matrix (3,3)
    """
    point_no = 8  # 8-point algorithm
    N = 10000 # number of iterations
    F_best = None
    min_error = np.inf
    max_inlier = 0
    threshold = 0.1  # threshold for inliers

    outlier_ratio = 0.65  # num of outliers / num of points
    
    # Adaptively determining the number of iterations
    if adaptive:
        p = 0.95  # Required probability of success
        N = np.log(1 - p) / np.log(1 - (pow(1 - outlier_ratio, point_no)))
        N = int(N)

    kps1_homo = np.hstack((kps1, np.ones((kps1.shape[0], 1))))
    kps2_homo = np.hstack((kps2, np.ones((kps1.shape[0], 1))))
    kp1_3 = np.tile(kps1_homo, 3)
    kp2_3 = kps2_homo.repeat(3, axis=1)
    A = np.multiply(kp1_3, kp2_3)

    for _ in range(N):
        indx = np.random.randint(kps1.shape[0], size=point_no)
        subset_1 = kps1[indx, :]
        subset_2 = kps2[indx, :]

        F_est = findFundamentalM(subset_1, subset_2)
        error = np.abs(A @ F_est.reshape((-1)))
        current_in = np.sum(error <= threshold)
        if current_in > max_inlier:
            F_best = F_est.copy()
            max_inlier = current_in

    final_error = np.abs(A @ F_best.reshape((-1)))
    indices_inliers = np.argsort(final_error)
    inliers_best = indices_inliers[:29] # return less than 30 points for uncluttered vis of epipolar lines
    mask = np.zeros(kps1.shape[0], dtype=int)
    mask[inliers_best] = 1
    return F_best, mask.reshape((-1,1))

def findEssentialM_RANSAC(kps1, kps2, K):
    """Estimate the essential matrix using RANSAC."""
    # max_inliers = 0
    # E_best = None
    # threshold = 0.1
    # iterations = 1000
    # inliers_best = []
    # for _ in range(iterations):
    #     # select points for the minimal set for 8-point algorithm
    #     indx = np.random.randint(kps1.shape[0], size=8) #np.random.choice(len(kps1), 8, replace=False)
    #     subset_1 = kps1[indx]
    #     subset_2 = kps2[indx]
        
    #     E = K.T @ findFundamentalM(subset_1, subset_2) @ K
        
    #     # Count inliers
    #     inlier_indices = []
    #     for i, (p1, p2) in enumerate(zip(kps1, kps2)):
    #         epipolar_const = p2 @ E @ (np.append(p1,1)) #np.dot(p2, np.dot(F, np.append(p1, 1)))
    #         if abs(epipolar_const) < threshold:
    #             inlier_indices.append(i)
        
    #     num_inliers = len(inlier_indices)
    #     if num_inliers > max_inliers:
    #         max_inliers = num_inliers
    #         E_best = E
    #         inliers_best = inlier_indices

    F_best, inliers_best = findFundamentalM_RANSAC(kps1, kps2)
   
    mask = np.zeros(kps1.shape[0], dtype=int)
    mask[inliers_best] = 1
        
    E_best = K.T @ F_best @ K
    
    return E_best, mask.reshape((-1,1))

def compute_homography(src_pts, dst_pts):
    # We need 4 pairs of points to compute the homography
    assert len(src_pts) == len(dst_pts) == 4, "There must be four pairs of points."

    A = []
    for i in range(4):
        x = src_pts[i][0]
        y = src_pts[i][1]
        u = dst_pts[i][0]
        v = dst_pts[i][1]
        # construct the matrix A
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])

    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    Hom = Vh[-1].reshape((3, 3))
    H = Hom / Hom[2, 2] # Normalize the homography matrix
    return H

def findHomography_RANSAC(kps1, kps2, max_iter=1000, threshold=1.0):
    """
    Estimate the homography matrix from point correspondences using RANSAC
    Input:
    kps1: keypoints of the first image (N,2) - source points
    kps2: keypoints of the second image (N,2) - destination points
    Output:
    H_best: the estimated homography matrix (3,3)
    best_inliers: the indices of the inliers
    """
    max_inliers = 0
    H_best = None
    best_inliers = []
    for _ in range(max_iter):
        indx = np.random.choice(len(kps1), 4, replace=False)
        subset_1 = kps1[indx]
        subset_2 = kps2[indx]

        H = compute_homography(subset_1, subset_2)

        # Compute and count inliers
        inlier_indices = []
        for i in range(len(kps1)):
            point_src = np.append(kps1[i], 1)
            point_dst = np.append(kps2[i], 1)
            estimated_dst = H @ point_src
            estimated_dst /= estimated_dst[2]  # homogeneous coords to cartesian coords

            error = np.linalg.norm(kps2[i] - estimated_dst[:2]) # np.sqrt(np.sum((kps2[i] - estimated_dst[:2]) ** 2), axis=0))
            if error < threshold:
                inlier_indices.append(i)

        num_inliers = len(inlier_indices)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            H_best = H
            best_inliers = inlier_indices

    # Use all inliers to compute the final homography
    # if best_inliers:
    #     all_inlier_src = kps1[best_inliers]
    #     all_inlier_dst = kps2[best_inliers]
    #     H_best = compute_homography(all_inlier_src, all_inlier_dst)
    mask = np.zeros(kps1.shape[0], dtype=int)
    mask[best_inliers] = 1
    return H_best, mask.reshape((-1,1))

def decomposeEssentialM(E):
    U, _, Vh = np.linalg.svd(E)
    
    # the constraint that E should have two singular values equal and one equal to zero
    S_new = np.diag([1, 1, 0])  
    E_new = np.dot(U, np.dot(S_new, Vh))  

    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # possible rot matrices
    Rot1 = U @ W @ Vh 
    Rot2 = U @ W.T @ Vh

    t = U[:, 2]  

    # check det(R) = 1 for proper rot matrices
    if np.linalg.det(Rot1) < 0:
        Rot1 = -Rot1
    if np.linalg.det(Rot2) < 0:
        Rot2 = -Rot2

    return Rot1, Rot2, t

def checkCheirality(P, R, t):
    """
    this is the condition of a 3D point being in front of both cameras 
    and is called 'cheirality' (https://www-users.cse.umn.edu/~hspark/CSci5980/hw4.pdf)
    Input:
    P: 3D points (N,3)
    R: rotation matrix (3,3)
    t: translation vector (3,)
    Output:
    count: the number of points in front of both cameras
    """
    count = 0
    for point3D in P:
        p_homo = np.append(point3D, 1)
        depth1 = p_homo[2]
        
        # project the point to 2nd cam
        p_cam2 = R @ point3D[:-1] + t 
        depth2 = p_cam2[2]

        if depth1 > 0 and depth2 > 0:
            count += 1
    return count

def triangulatePoints(proj1, proj2, kps1, kps2):
    """
    Triangulate the points
    Input:
    proj1: projection matrix of the first camera (3,4)
    proj2: projection matrix of the second camera (3,4)
    kps1: keypoints of the first image (N,2)
    kps2: keypoints of the second image (N,2)
    Output:
    triangulated_points: the triangulated points (N,4)
    """
    kps1 = kps1.T
    kps2 = kps2.T
    num_points = kps1.shape[1]

    triangulated_points = np.zeros((num_points, 4))

    for i in range(num_points):
        A = np.zeros((4, 4))
        A[0, :] = kps1[0, i] * proj1[2, :] - proj1[0, :]
        A[1, :] = kps1[1, i] * proj1[2, :] - proj1[1, :]
        A[2, :] = kps2[0, i] * proj2[2, :] - proj2[0, :]
        A[3, :] = kps2[1, i] * proj2[2, :] - proj2[1, :]

        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1] # take the vector corresponding to the smallest singular value
        X /= X[-1]
        triangulated_points[i, :] = X

    return triangulated_points

def recoverPose(E, kps1, kps2, K):
    R1, R2, t = decomposeEssentialM(E)
    possible_rot_trans = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    max_inliers = -1
    R_best = None
    t_best = None    
   
    for R, t in possible_rot_trans:
        # triangulate points for cheirality check
        P1 = np.eye(3, 4)
        # make projection matrix from R and t
        P2 = np.hstack((R, t.reshape(3, 1)))
        points3D = triangulatePoints(P1, P2, kps1, kps2)
        count = checkCheirality(points3D, R, t)
        if count > max_inliers:
            max_inliers = count
            R_best = R
            t_best = -t # changed according to the result from opencv
    return R_best, t_best.reshape((3,1)), max_inliers

def Rodrigues(r):
    """
    Convert the rotation vector to the rotation matrix
    Input:
    r: the rotation vector (3,)
    Output:
    R: the rotation matrix (3,3)
    Formulas from https://github.com/personalrobotics/tsr/blob/2b8d9e12c6dc2eb20b47bced6e416cf09a00e5ac/src/tsr/rodrigues.py#L34
    """
    theta = np.linalg.norm(r)
    if theta > 1e-30:
        r_norm = r / theta
        Sn = np.array([[0, float(-r_norm[2]), float(r_norm[1])],
                      [float(r_norm[2]), 0, float(-r_norm[0])],
                      [float(-r_norm[1]), float(r_norm[0]), 0]])
        R = np.eye(3) + np.sin(theta) * Sn + (1 - np.cos(theta)) * (Sn @ Sn)
    else:
        Sr = np.array([[0, -r[2][0], r[1][0]],
                      [r[2][0], 0, -r[0][0]],
                      [-r[1][0], r[0][0], 0]])
        theta_2 = theta ** 2
        R = np.eye(3) + (1 - theta_2 / 6.0) * Sr + (0.5 - theta_2 / 24.0) * (Sr @ Sr)
    return R    

def projectPoints(points3D, r, t, K):
    """
    Project 3D points to 2D image points.
    Input:
    points3D (N,3): 3D points.
    r: rotation vector (3,)
    t: translation vector (3,)
    K: intrinsic parameters.
    """
    if r.ndim == 1:
        R = Rodrigues(r)
    else:
        R = r

    Rt = np.hstack((R, t.reshape(3, 1)))

    if points3D.shape[1] == 3:
        points3D = np.hstack((points3D, np.ones((points3D.shape[0], 1))))

    points3D_camcoord = (Rt @ points3D.T).T
    points2D_homo = (K @ points3D_camcoord.T).T
    points2D_pixcoords = points2D_homo[:, :2] / points2D_homo[:, 2][:, np.newaxis]

    return points2D_pixcoords

def solvePnP(points3D, points2D, K):
    obj_pts_homo = np.hstack((points3D, np.ones((points3D.shape[0], 1))))
    num_points = points3D.shape[0]

    A = np.zeros((2 * num_points, 12))
    for i in range(num_points):
        X = obj_pts_homo[i, :]
        u, v = points2D[i, :]
        A[2*i] = np.array([-X[0], -X[1], -X[2], -1, 0, 0, 0, 0, u*X[0], u*X[1], u*X[2], u])
        A[2*i + 1] = np.array([0, 0, 0, 0, -X[0], -X[1], -X[2], -1, v*X[0], v*X[1], v*X[2], v])
    
    try:
        _, _, Vt = np.linalg.svd(A)
        success = True
    except np.linalg.LinAlgError:  # potential SVD convergence failure
        print('SVD did not converge')
        success = False
        return np.eye(3), np.array([0.0,0.0,0.0])

    P = Vt[-1].reshape((3, 4))

    K = P[:, :3]  # left 3 by 3 part of P is the cam matrix (K?)
    inverse_K = np.linalg.inv(K)
    R, _ = np.linalg.qr(inverse_K)  # QR decompose
    t = inverse_K @ P[:, 3]

    return success, R, t    

def solvePnPRansac(points3D, points2D, K, iter=1000, reprojection_error=8.0):
    inliers_best = -1
    R_best = None
    t_best = None

    num_points = len(points3D)
    for _ in range(iter):
        indx = np.random.choice(num_points, 4, replace=False) # minimal set for PnP is 4
        subset_3Dpoints = points3D[indx]
        subset_2Dpoints = points2D[indx]

        success, R, t = solvePnP(subset_3Dpoints, subset_2Dpoints, K)
        if success is False:
            continue

        projected_2Dpoints = projectPoints(points3D, R, t, K)
        reprojection_errors = np.linalg.norm(projected_2Dpoints.squeeze() - points2D, axis=1)
        count_inliers = np.sum(reprojection_errors < reprojection_error)

        if count_inliers > inliers_best:
            inliers_best = count_inliers
            R_best = R
            t_best = t

    # solve PnP with all inliers 
    if inliers_best > 0:
        inlier_mask = reprojection_errors < reprojection_error
        final_obj_pts = points3D[inlier_mask]
        final_img_pts = points2D[inlier_mask]
        success, R, t = solvePnP(final_obj_pts, final_img_pts, K)
        if success:
            R_best = R
            t_best = t

    return R_best, t_best, inliers_best

    




