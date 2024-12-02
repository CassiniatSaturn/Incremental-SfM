## Pipeline
+ Ref: https://cmsc426.github.io/sfm/
1. Feature Matching and Outlier rejection using RANSAC
2. Estimating Fundamental Matrix
3. Estimating Essential Matrix from Fundamental Matrix
4. Estimate Camera Pose from Essential Matrix
5. Check for Cheirality Condition using Triangulation
6. Perspective-n-Point
7. Bundle Adjustment
## Code Structure
+ data: store provided data, colmap_gt is the estimated results using COLMAP
+ estimation: stores predcition in json files
+ evaluation: provided evaluation scripts
+ src
    + sfm_cv2.py: implementation with OpenCV and Pytorch for validate
    + sfm.py: implementation from scratch
        + ref: https://github.com/aadhyap/Structure-From-Motion_NeRF/tree/2ea5f9c0c6e237d41373cb84c13ce5c76b0ed432
    + utils.py: some functions to handle read, write files
## Remark
+ Notations following lecture slides
+ Eavalutation on estimation from COLMAP
    + box pose: Rotation error: 1.92,Translation error: 0.0
    + box point: 3.1208732408230317 CD

## Reference
+ https://www.cs.cmu.edu/afs/andrew/scs/cs/15-463/f07/proj_final/www/amichals/fundamental.pdf
