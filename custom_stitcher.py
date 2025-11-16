import cv2
import numpy as np
from typing import List, Tuple

def build_gaussian_pyramid(img, levels):
    G = img.copy()
    gp = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp

def build_laplacian_pyramid(gp, levels):
    lp = [gp[levels-1]]
    for i in range(levels-1, 0, -1):
        # cv2.pyrUp yêu cầu kích thước đích
        size = gp[i-1].shape[:2]
        GE = cv2.pyrUp(gp[i], dstsize=(size[1], size[0]))
        L = cv2.subtract(gp[i-1], GE)
        lp.append(L)
    return lp
# --- Hỗ trợ scale ảnh ---
def resize_for_work(image, scale: float):
    if scale == 1.0:
        return image
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

# --- Detect + describe bằng ORB ---
def detect_and_describe(image_gray, nfeatures=5000):
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
# Thay thế hàm warp_and_blend cũ

def warp_and_blend(A, B, H, levels=5):
    """
    Thực hiện Warping và Blending bằng Kim tự tháp Laplace.
    """
    hA, wA = A.shape[:2]
    hB, wB = B.shape[:2]
    
    # 1. TÍNH TOÁN KHUNG HÌNH (CANVAS)
    # Corners của B sau khi warp
    cornersB = np.float32([[0,0],[wB,0],[wB,hB],[0,hB]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(cornersB, H)
    all_corners = np.concatenate((warped_corners, np.float32([[0,0],[wA,0],[wA,hA],[0,hA]]).reshape(-1,1,2)))
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    translation = [-xmin, -ymin]
    T = np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])
    
    canvas_size = (xmax - xmin, ymax - ymin)
    
    # 2. WARP CẢ HAI ẢNH VỀ KHUNG HÌNH CHUNG
    # Warp B
    warped_B = cv2.warpPerspective(B, T @ H, canvas_size)
    # Tạo ảnh A trên canvas
    warped_A = np.zeros_like(warped_B)
    warped_A[translation[1]:translation[1]+hA, translation[0]:translation[0]+wA] = A
    
    # 3. TẠO MẶT NẠ (MASK)
    # Tạo mask để hòa trộn (mask này sẽ xác định vùng chồng lấp)
    mask_A = np.zeros((hA, wA), dtype=np.uint8)
    mask_A[:,:] = 255
    warped_mask_A = cv2.warpPerspective(mask_A, T, canvas_size)
    
    mask_B = np.zeros((hB, wB), dtype=np.uint8)
    mask_B[:,:] = 255
    warped_mask_B = cv2.warpPerspective(mask_B, T @ H, canvas_size)
    
    # 4. CHUYỂN ĐỔI SANG FLOAT (CẦN THIẾT CHO BLENDING ĐA DẢI)
    warped_A = warped_A.astype(np.float32)
    warped_B = warped_B.astype(np.float32)

    # 5. LAPLACIAN PYRAMID BLENDING
    
    # Số cấp độ
    min_dim = min(canvas_size)
    levels = int(np.floor(np.log2(min_dim) - 2)) # Giới hạn cấp độ hợp lý
    
    # 5a. Xây dựng Kim tự tháp cho A, B và Mask
    gp_A = build_gaussian_pyramid(warped_A, levels)
    gp_B = build_gaussian_pyramid(warped_B, levels)
    lp_A = build_laplacian_pyramid(gp_A, levels)
    lp_B = build_laplacian_pyramid(gp_B, levels)

    # Xây dựng Kim tự tháp Gaussian cho Blending Mask (Mặt nạ Gaussian)
    # Mask hòa trộn (Weighted Mask)
    # Tạo mask M: 0 ở vùng A, 1 ở vùng B, gradient ở giữa
    M = warped_mask_B.copy().astype(np.float32)
    
    # Tìm vùng chồng lấp (vùng overlap)
    overlap = cv2.bitwise_and(warped_mask_A, warped_mask_B)
    
    # Tạo gradient cho mask trong vùng overlap
    # Tính tâm vùng overlap
    rows, cols = np.where(overlap > 0)
    if len(rows) > 0:
        center_x = int((np.min(cols) + np.max(cols)) / 2)
        # Tạo gradient mask: mask_B sẽ giảm dần từ 1 về 0 tại trung tâm overlap
        M[:, center_x:] = 1.0 # B là ảnh mới, nằm bên phải, nên gradient sẽ từ phải sang trái
        M[:, :center_x] = 0.0 # A là ảnh cũ, nằm bên trái
        
        # Áp dụng smooth transition
        transition_width = min(500, int((np.max(cols) - np.min(cols)) / 2)) # Width of feathering
        
        if transition_width > 0:
            for c in range(center_x - transition_width, center_x + transition_width):
                if c >= np.min(cols) and c <= np.max(cols):
                    # Linear feathering trong vùng overlap
                    M[:, c] = (c - (center_x - transition_width)) / (2 * transition_width)
        
    gp_M = build_gaussian_pyramid(M, levels)
    
    # 5b. Hòa trộn từng cấp độ
    LS = []
    for i in range(levels):
        # Lớp thứ i = (L_A * (1 - M_i)) + (L_B * M_i)
        # M_i là Gaussian Pyramid của Mask
        
        # Đảm bảo kích thước khớp, vì M_i là float, L_A, L_B là float
        M_i = cv2.resize(gp_M[levels - 1 - i], lp_A[i].shape[:2][::-1]) 
        
        # Mở rộng M_i thành 3 kênh để nhân với ảnh màu
        M_i_3ch = np.expand_dims(M_i, axis=2)
        
        L = (lp_A[i] * (1.0 - M_i_3ch)) + (lp_B[i] * M_i_3ch)
        LS.append(L)

    # 5c. Tái tạo ảnh kết quả
    blended = LS[0]
    for i in range(1, levels):
        # cv2.pyrUp yêu cầu kích thước đích
        size = LS[i].shape[:2]
        blended = cv2.pyrUp(blended, dstsize=(size[1], size[0]))
        blended = cv2.add(blended, LS[i])

    # 6. DÁN CÁC VÙNG KHÔNG CHỒNG LẤP (NẾU CÓ)
    # Lấy vùng B không chồng lấp
    non_overlap_B = cv2.subtract(warped_B, cv2.bitwise_and(warped_B, cv2.merge([warped_mask_A] * 3).astype(np.float32)))
    # Dán lên ảnh hòa trộn
    blended = cv2.add(blended, non_overlap_B)
    
    # Lấy vùng A không chồng lấp
    non_overlap_A = cv2.subtract(warped_A, cv2.bitwise_and(warped_A, cv2.merge([warped_mask_B] * 3).astype(np.float32)))
    # Dán lên ảnh hòa trộn
    blended = cv2.add(blended, non_overlap_A)
    
    # Trả về kết quả dưới dạng uint8
    return np.clip(blended, 0, 255).astype(np.uint8)
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
