import cv2
import numpy as np

# Step 1: Estimate the Fundamental Matrix

# You can estimate the fundamental matrix using feature points from both images. The fundamental matrix relates corresponding points in two images when the cameras’ intrinsic parameters are unknown or not used.
# Step 2: Compute the Essential Matrix

# If you have an estimate of the intrinsic parameters, you can convert the fundamental matrix to the essential matrix, which then can be decomposed into possible rotation and translation matrices.
# Step 3: Decompose the Essential Matrix

# From the essential matrix, you can extract the rotation and translation up to scale. You’ll get four possible solutions, and you will need to choose the correct one based on additional constraints or testing.
# Step 4: Triangulate Points

# Using the chosen rotation and translation matrices along with the intrinsic parameters, triangulate the 3D positions of matched points.

# Here is how you could implement this in Python using OpenCV:

# Load your images
img1 = cv2.imread("IMG_240412_084811_0130_GRE.TIF", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("IMG_240412_084811_0131_GRE.TIF", cv2.IMREAD_GRAYSCALE)

# Detect keypoints and descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match descriptors
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract the matched points
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Compute the fundamental matrix
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Assuming you have the camera matrix K
focal_length_px = 1661.2583368971314  # focal length in pixels
px_p = 611.8027223624858  # principal point x-coordinate in pixels
py_p = 467.06350970174316  # principal point y-coordinate in pixels

# Constructing the intrinsic matrix K
K = np.array([[focal_length_px, 0, px_p], [0, focal_length_px, py_p], [0, 0, 1]])

# Compute the essential matrix
E = K.T @ F @ K

# Decompose the essential matrix into R and t
retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

# Triangulation
# Projection matrix for the first camera
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
# Projection matrix for the second camera
P2 = np.hstack((R, t))

# Convert points to homogeneous coordinates
pts1_h = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
pts2_h = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]

# Triangulate points
points4D = cv2.triangulatePoints(P1, P2, pts1_h.T, pts2_h.T)

# Convert from homogeneous to 3D coordinates
points3D = points4D[:3] / points4D[3]

# Print or use the 3D points
print("3D Points:")
print(points3D.T)
