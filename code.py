import cv2 as cv
import os
import numpy as np
from tqdm import tqdm

def generate_feature_matching_video(image_folder, output_video_path, num_frames=None):

    # Get sorted image paths
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if num_frames:
        image_files = image_files[:num_frames]

    # Read the first image to determine frame size
    first_img_path = os.path.join(image_folder, image_files[0])
    first_img = cv.imread(first_img_path, cv.IMREAD_GRAYSCALE)
    height, width = first_img.shape

    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # FLANN matcher setup
    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv.FlannBasedMatcher(indexParams, searchParams)

    # For video
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = 20  # have tested with different fps
    match_height, match_width = height, width * 2
    out = cv.VideoWriter(output_video_path, fourcc, fps, (match_width, match_height))

    print(f"Processing {len(image_files) - 1} image pairs...")

    for i in tqdm(range(len(image_files) - 1)):
        img1_path = os.path.join(image_folder, image_files[i])
        img2_path = os.path.join(image_folder, image_files[i + 1])

        img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

        img1_color = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        img2_color = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

        keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptor2 = sift.detectAndCompute(img2, None)

        if descriptor1 is not None and descriptor2 is not None:
            matches = flann.knnMatch(descriptor1, descriptor2, k=2)

            # Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance: #based on the original SIFT paper
                    good_matches.append(m)

            # Prepare data for RANSAC
            if len(good_matches) >= 8: #need at least 4 point pairs to compute a homography
                pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

                # RANSAC to filter inliers
                H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0) #reprojection threshold in pixels, if the error is less than or equal to 5, it is considered inlier
                inlier_matches = [good_matches[j] for j in range(len(mask)) if mask[j]]

                # Draw only inliers
                match_img = cv.drawMatches(img1, keypoints1, img2, keypoints2,
                                           inlier_matches, None,
                                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            else:
                match_img = np.zeros((match_height, match_width, 3), dtype=np.uint8)
                cv.putText(match_img, "Not enough good matches for RANSAC", (50, height // 2),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            match_img = np.zeros((match_height, match_width, 3), dtype=np.uint8)
            cv.putText(match_img, "Descriptor extraction failed", (50, height // 2),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(match_img)

    out.release()
    print(f"Video saved to {output_video_path}")


# For the project's image feature tracking: 
if __name__ == "__main__":
    image_folder = r'path/to/your/folder'
    output_video = r'path/to/your/folder/demo_video.mp4'
    generate_feature_matching_video(image_folder, output_video, num_frames=200)
