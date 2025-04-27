# Feature-Tracking-on-200-Images
This project implements feature tracking over a sequence of 200 images using the SIFT (Scale-Invariant Feature Transform) algorithm.

## Objective
To visualize matching features across frames in a video with tracking accuracy and speed. The video output displays matches between adjacent frames, where only geometrically valid inliers (using RANSAC) are shown.


## Features and Algorithms Used

- Feature Detector & Descriptor:  
  [SIFT] — A scale and rotation-invariant feature detector

- Feature Matching Algorithm:  
  FLANN (Fast Library for Approximate Nearest Neighbors) — efficient for high-dimensional feature matching

- Outlier Rejection:  
  Lowe’s Ratio Test (threshold = 0.75) — filters ambiguous matches  
  RANSAC-based Homography Estimation — removes spatially inconsistent matches

- Video Output:  
  A side-by-side visualization of adjacent frames with correct feature matches (inliers only)  
  Frame Rate: 20 FPS (for smoother and faster playback)

## How It Works
1. Load all image frames in sorted order
2. For each pair of consecutive frames:
   - Detect SIFT keypoints and compute descriptors
   - Match descriptors using FLANN with k=2 nearest neighbors
   - Apply Lowe's ratio test to retain high-confidence matches
   - Use RANSAC to compute a homography matrix and filter out incorrect matches
   - Draw only inlier matches for visualization
3. Combine matching images into a video using OpenCV’s `VideoWriter` function

## How to Run
Requirements:
- Python 3.6+
- OpenCV (with SIFT support)
- tqdm

Also, change the address of image_folder and output_video according to your PC's addresses in code.py file.
