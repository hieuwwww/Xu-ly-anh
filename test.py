import cv2
from Stitcher import Stitcher
from org_stitcher import (
    load_images,
    stitch_images,
    crop_black_borders,
    save_image,
)

INPUT_ROOT = "unstichedImages"
OUTPUT_ROOT = "panoramaImage"


# Đọc ảnh
patterns = [f'{INPUT_ROOT}/*.png', f'{INPUT_ROOT}/*.jpg']
images = []
for p in patterns:
    images += load_images(p)

# Tạo stitcher
stitcher = Stitcher.create(Stitcher.PANORAMA)
# stitcher.setCompositingResol(1.0)  # tùy chọn

# Stitch
status, pano = stitcher.stitch(images)

if status == Stitcher.Status.OK:
    cv2.imwrite("panorama.jpg", pano)
    cv2.imshow("Panorama", pano)
    cv2.waitKey(0)
else:
    print("Stitching failed!", status)