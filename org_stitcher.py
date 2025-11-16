import glob
from typing import List, Tuple

import cv2
import imutils
import numpy as np


def load_images(pattern: str) -> List[np.ndarray]:
    paths = glob.glob(pattern)
    images: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"Warning: không đọc được ảnh: {p}")
            continue
        images.append(img)
    return images


def load_images_from_files(paths: List[str]) -> List[np.ndarray]:
    """Tải ảnh từ danh sách đường dẫn cụ thể (nếu file không tồn tại hoặc đọc lỗi sẽ bỏ qua)."""
    images: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"Warning: không đọc được ảnh: {p}")
            continue
        images.append(img)
    return images


def stitch_images(images: List[np.ndarray]) -> Tuple[int, np.ndarray]:
    """Ghép các ảnh lại với nhau.

    Trả về (status, stitched_image). status == 0 nghĩa là thành công.
    """
    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch(images)
    return status, pano


def _largest_rectangle_in_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Tìm hình chữ nhật trục song song lớn nhất hoàn toàn nằm trong mask (mask=1 là vùng hợp lệ).

    Trả về (top, bottom, left, right). Thuật toán: largest rectangle in binary matrix
    bằng histogram + stack cho từng hàng.
    """
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best = (0, h - 1, 0, w - 1)

    for i in range(h):
        row = mask[i] > 0
        # cập nhật histogram
        heights = heights + row.astype(np.int32)
        heights[~row] = 0

        # largest rectangle trong histogram hiện tại
        stack: List[Tuple[int, int]] = []  # (start_index, height)
        for j in range(w + 1):
            cur = heights[j] if j < w else 0
            start = j
            while stack and stack[-1][1] > cur:
                idx, hgt = stack.pop()
                start = idx
                area = hgt * (j - idx)
                if area > best_area:
                    best_area = area
                    top = i - hgt + 1
                    bottom = i
                    left = idx
                    right = j - 1
                    best = (top, bottom, left, right)
            stack.append((start, cur))

    return best


def crop_black_borders(stitched: np.ndarray, margin: int = 6, close_iters: int = 2) -> np.ndarray:
    """Cắt phần biên đen bằng cách tìm hình chữ nhật lớn nhất nằm trong vùng ảnh hợp lệ.

    Bước:
    - Thêm viền nhỏ tránh cắt sát mép.
    - Tạo mask nhị phân (pixel > 0 là hợp lệ) và đóng morphological để lấp lỗ.
    - Tìm hình chữ nhật lớn nhất hoàn toàn nằm trong mask (largest rectangle in binary matrix).
    - Cắt theo hình chữ nhật đó, cộng thêm margin nhỏ.
    """
    # Thêm viền nhỏ để tránh mất mép khi cắt
    stitched = cv2.copyMakeBorder(stitched, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

    top, bottom, left, right = _largest_rectangle_in_mask(bw)

    # margin nhỏ để không cắt sát, và giữ trong biên ảnh
    y1 = max(top - margin, 0)
    y2 = min(bottom + margin, stitched.shape[0] - 1)
    x1 = max(left - margin, 0)
    x2 = min(right + margin, stitched.shape[1] - 1)

    cropped = stitched[y1:y2 + 1, x1:x2 + 1]
    return cropped


def save_image(path: str, img: np.ndarray) -> None:
    """Ghi ảnh ra file. Tạo thư mục cha nếu chưa tồn tại.

    Path có thể là đường dẫn tương đối hoặc tuyệt đối. Hàm sẽ đảm bảo
    thư mục đích tồn tại trước khi ghi file.
    """
    import os

    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    cv2.imwrite(path, img)


def show_image(title: str, img: np.ndarray, wait_ms: int = 0) -> None:
    """Hiện ảnh với OpenCV và đợi `wait_ms` mili giây (0 là chờ phím nhấn)."""
    cv2.imshow(title, img)
    cv2.waitKey(wait_ms)
