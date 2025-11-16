import os
from stitcher import (
    load_images,
    stitch_images,
    crop_black_borders,
    save_image,
)
from custom_stitcher import stitch as custom_stitch

INPUT_ROOT = "unstichedImages"
OUTPUT_ROOT = "panoramaImage"


def run_default_stitcher(images):
    """
    Chạy stitcher mặc định của OpenCV.
    Trả về (status, pano).
    """
    print("=== Stitcher mặc định OpenCV ===")
    status, pano = stitch_images(images)
    if status != 0:
        print(f"[Default Stitcher] Lỗi ghép: {status}")
        return None
    return pano


def run_custom_stitcher(images):
    """
    Chạy custom stitcher do bạn tự triển khai.
    Trả về ảnh panorama (hoặc None nếu lỗi).
    """
    print("=== Custom Stitcher ===")
    pano = custom_stitch(images)
    if pano is None:
        print("[Custom Stitcher] Ghép thất bại.")
        return None
    return pano


def main():
    # ⚡ Load ảnh
    patterns = [f'{INPUT_ROOT}/*.png', f'{INPUT_ROOT}/*.jpg']
    images = []
    for p in patterns:
        images += load_images(p)

    print(f"Tìm được {len(images)} ảnh trong thư mục gốc.")
    if len(images) < 2:
        print("Cần ≥ 2 ảnh để ghép.")
        return

    # ⚡ Chạy Stitcher mặc định
    pano_default = run_default_stitcher(images)
    if pano_default is not None:
        save_image(f"{OUTPUT_ROOT}/pano_default_raw.png", pano_default)
        cropped = crop_black_borders(pano_default, margin=8)
        save_image(f"{OUTPUT_ROOT}/pano_default_processed.png", cropped)

    # ⚡ Chạy Custom Stitcher
    pano_custom = run_custom_stitcher(images)
    if pano_custom is not None:
        save_image(f"{OUTPUT_ROOT}/pano_custom_raw.png", pano_custom)
        cropped = crop_black_borders(pano_custom, margin=8)
        save_image(f"{OUTPUT_ROOT}/pano_custom_processed.png", cropped)

    print("=== Hoàn tất: đã tạo 2 kết quả để so sánh ===")
    print(f"Kết quả xem tại: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
