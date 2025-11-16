import cv2

import numpy as np

from typing import List, Tuple



# --- Hỗ trợ scale ảnh ---

def resize_for_work(image, scale: float):

    if scale == 1.0:

        return image

    h, w = image.shape[:2]

    return cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)



# --- Detect + describe bằng ORB ---

def detect_and_describe(image_gray, nfeatures=8000):

    orb = cv2.ORB_create(nfeatures)

    keypoints, descriptors = orb.detectAndCompute(image_gray, None)

    return keypoints, descriptors



# --- Match features bằng BFMatcher + ratio test ---

def match_features(descA, descB, ratio=0.75):

    if descA is None or descB is None:

        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    raw_matches = matcher.knnMatch(descA, descB, k=2)

    good = []

    for m,n in raw_matches:

        if m.distance < ratio * n.distance:

            good.append(m)

    return good



# --- Estimate homography ---

def estimate_homography(kpsA, kpsB, matches):

    if len(matches) < 4:

        return None

    ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])

    ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

    H, _ = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 3.0)

    return H



# --- Warp và blend (feathering đơn giản) ---

def warp_and_blend(A, B, H):

    hA, wA = A.shape[:2]

    hB, wB = B.shape[:2]

    # corners of B after warp

    cornersB = np.float32([[0,0],[wB,0],[wB,hB],[0,hB]]).reshape(-1,1,2)

    warped_corners = cv2.perspectiveTransform(cornersB, H)

    all_corners = np.concatenate((warped_corners, np.float32([[0,0],[wA,0],[wA,hA],[0,hA]]).reshape(-1,1,2)))

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)

    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-xmin, -ymin]

    T = np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])

    # warp B

    result = cv2.warpPerspective(B, T @ H, (xmax - xmin, ymax - ymin))

    # paste A

    result[translation[1]:translation[1]+hA, translation[0]:translation[0]+wA] = A

    return result



# --- Tính scale làm việc ---

def compute_work_scale(images, target_max_pixels=1e6):

    scales = []

    for img in images:

        h,w = img.shape[:2]

        scale = min(1.0, np.sqrt(target_max_pixels / (h*w)))

        scales.append(scale)

    return scales



# --- Main stitch function ---

def stitch(images: List[np.ndarray], target_max_pixels=1e6) -> Tuple[int, np.ndarray]:

    if len(images) < 2:

        return 1, None



    # tính scale để giảm RAM

    scales = compute_work_scale(images, target_max_pixels)



    # scale ảnh cho feature detection

    gray_scaled = [cv2.cvtColor(resize_for_work(img, s), cv2.COLOR_BGR2GRAY)

                   for img,s in zip(images, scales)]

    keypoints = []

    descriptors = []

    for gray in gray_scaled:

        kps, desc = detect_and_describe(gray)

        keypoints.append(kps)

        descriptors.append(desc)



    pano = images[0]

    for i in range(1, len(images)):

        matches = match_features(descriptors[i-1], descriptors[i])

        if len(matches) < 4:

            return 2, None

        # map back to full-res coordinates

        sA = scales[i-1]; sB = scales[i]

        kpsA_full = [cv2.KeyPoint(k.pt[0]/sA, k.pt[1]/sA, k.size) for k in keypoints[i-1]]

        kpsB_full = [cv2.KeyPoint(k.pt[0]/sB, k.pt[1]/sB, k.size) for k in keypoints[i]]

        H = estimate_homography(kpsA_full, kpsB_full, matches)

        if H is None:

            return 3, None

        pano = warp_and_blend(pano, images[i], H)

    return pano