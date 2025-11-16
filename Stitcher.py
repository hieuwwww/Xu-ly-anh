# Stitcher.py
import cv2
import numpy as np
from typing import List, Optional, Tuple

class Stitcher:
    PANORAMA = 0
    SCANS = 1

    class Status:
        OK = 0
        ERR_NEED_MORE_IMGS = 1
        ERR_HOMOGRAPHY_EST_FAIL = 2
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3

    ORIG_RESOL = -1.0

    def __init__(self, mode=PANORAMA):
        self.mode = mode
        self.registr_resol = 0.6
        self.seam_est_resol = 0.1
        self.compose_resol = self.ORIG_RESOL
        self.pano_conf_thresh = 1.0
        self.interp_flags = cv2.INTER_LINEAR
        self.wave_correction = mode == self.PANORAMA
        self.wave_correct_kind = cv2.detail.WAVE_CORRECT_HORIZ

        self.features_finder = cv2.ORB_create()
        self.features_matcher = None
        self.estimator = None
        self.bundle_adjuster = None
        self.warper_creator = None  # Factory cho PyRotationWarper
        self.exposure_comp = None
        self.seam_finder = None
        self.blender = None

        self.work_scale = self.seam_scale = self.seam_work_aspect = 1.0
        self.warped_image_scale = 1.0
        self.imgs = self.features = self.cameras = self.indices = []
        self.seam_est_imgs = self.full_img_sizes = []

        self._setup_components()

    def _setup_components(self):
        if self.mode == self.PANORAMA:
            self.estimator = cv2.detail.HomographyBasedEstimator()
            self.features_matcher = cv2.detail.BestOf2NearestMatcher(False)
            self.bundle_adjuster = cv2.detail.BundleAdjusterRay()
            # Dùng PyRotationWarper thay vì SphericalWarper (không tồn tại)
            self.warper_creator = lambda scale: cv2.PyRotationWarper("spherical", scale)
            self.exposure_comp = cv2.detail.BlocksGainCompensator()
            self.seam_finder = cv2.detail.GraphCutSeamFinder('COST_COLOR')
            self.blender = cv2.detail.MultiBandBlender(0)  # SỬA: 0 thay vì False

        elif self.mode == self.SCANS:
            self.estimator = cv2.detail.AffineBasedEstimator()
            self.features_matcher = cv2.detail.AffineBestOf2NearestMatcher(False, False)
            self.bundle_adjuster = cv2.detail.BundleAdjusterAffinePartial()
            # Dùng PyRotationWarper cho affine
            self.warper_creator = lambda scale: cv2.PyRotationWarper("affine", scale)
            self.exposure_comp = cv2.detail.NoExposureCompensator()
            self.seam_finder = cv2.detail.GraphCutSeamFinder('COST_COLOR')
            self.blender = cv2.detail.MultiBandBlender(0)  # SỬA: 0 thay vì False
            self.wave_correction = False
        else:
            raise ValueError("Invalid mode")

    @staticmethod
    def create(mode=PANORAMA):
        return Stitcher(mode)

    def stitch(self, images: List[np.ndarray], masks: Optional[List[np.ndarray]] = None) -> Tuple[int, np.ndarray]:
        status = self.estimateTransform(images, masks)
        if status != self.Status.OK:
            return status, None
        return self.composePanorama()

    def estimateTransform(self, images: List[np.ndarray], masks: Optional[List[np.ndarray]] = None) -> int:
        self.imgs = [img.copy() for img in images]
        self.masks = [m.copy() for m in (masks or [])]
        if len(self.imgs) < 2:
            return self.Status.ERR_NEED_MORE_IMGS

        status = self._matchImages()
        if status != self.Status.OK:
            return status
        return self._estimateCameraParams()

    def composePanorama(self) -> Tuple[int, np.ndarray]:
        return self._composePanorama()

    def _matchImages(self) -> int:
        if len(self.imgs) < 2:
            return self.Status.ERR_NEED_MORE_IMGS

        self.work_scale = self.seam_scale = self.seam_work_aspect = 1.0
        is_work_set = is_seam_set = False
        self.features = []
        self.seam_est_imgs = []
        self.full_img_sizes = [(img.shape[1], img.shape[0]) for img in self.imgs]

        work_imgs = []
        for i, img in enumerate(self.imgs):
            h, w = img.shape[:2]
            area = w * h

            if self.registr_resol >= 0 and not is_work_set:
                self.work_scale = min(1.0, np.sqrt(self.registr_resol * 1e6 / area))
                is_work_set = True
            work_img = cv2.resize(img, None, fx=self.work_scale, fy=self.work_scale, interpolation=cv2.INTER_LINEAR)
            work_imgs.append(work_img)

            if not is_seam_set:
                self.seam_scale = min(1.0, np.sqrt(self.seam_est_resol * 1e6 / area))
                self.seam_work_aspect = self.seam_scale / self.work_scale
                is_seam_set = True
            seam_img = cv2.resize(img, None, fx=self.seam_scale, fy=self.seam_scale, interpolation=cv2.INTER_LINEAR)
            self.seam_est_imgs.append(seam_img)

        self.features = cv2.detail.computeImageFeatures2(self.features_finder, work_imgs)
        self.pairwise_matches = self.features_matcher.apply2(self.features)
        self.features_matcher.collectGarbage()

        self.indices = cv2.detail.leaveBiggestComponent(self.features, self.pairwise_matches, self.pano_conf_thresh)

        self.imgs = [self.imgs[i] for i in self.indices]
        self.seam_est_imgs = [self.seam_est_imgs[i] for i in self.indices]
        self.full_img_sizes = [self.full_img_sizes[i] for i in self.indices]

        return self.Status.OK if len(self.imgs) >= 2 else self.Status.ERR_NEED_MORE_IMGS

    def _estimateCameraParams(self) -> int:
        self.cameras = []  # Khởi tạo nếu chưa có
        if not self.estimator.apply(self.features, self.pairwise_matches, self.cameras):
            return self.Status.ERR_HOMOGRAPHY_EST_FAIL

        self.bundle_adjuster.setConfThresh(self.pano_conf_thresh)
        if not self.bundle_adjuster.apply(self.features, self.pairwise_matches, self.cameras):
            return self.Status.ERR_CAMERA_PARAMS_ADJUST_FAIL

        focals = sorted(cam.focal for cam in self.cameras)
        mid = len(focals) // 2
        self.warped_image_scale = focals[mid] if len(focals) % 2 else (focals[mid-1] + focals[mid]) * 0.5

        if self.wave_correction:
            rmats = [cam.R for cam in self.cameras]
            rmats = cv2.detail.waveCorrect(rmats, self.wave_correct_kind)
            for i, cam in enumerate(self.cameras):
                cam.R = rmats[i]

        return self.Status.OK

    def _composePanorama(self) -> Tuple[int, np.ndarray]:
        if len(self.imgs) == 0:
            return self.Status.ERR_NEED_MORE_IMGS, None

        area = self.full_img_sizes[0][0] * self.full_img_sizes[0][1]
        compose_scale = 1.0
        if self.compose_resol > 0:
            compose_scale = min(1.0, np.sqrt(self.compose_resol * 1e6 / area))
        compose_work_aspect = compose_scale / self.work_scale
        warp_scale = self.warped_image_scale * compose_work_aspect

        # Tạo warper bằng PyRotationWarper
        warper = self.warper_creator(warp_scale)

        corners, sizes, masks_warped, images_warped = [], [], [], []

        for i, img in enumerate(self.seam_est_imgs):
            K = self.cameras[i].K().astype(np.float32)
            K[:2, :] *= self.seam_work_aspect

            corner, img_wp = warper.warp(img, K, self.cameras[i].R, self.interp_flags, cv2.BORDER_REFLECT)
            size = (img_wp.shape[1], img_wp.shape[0])

            mask = np.full(img.shape[:2], 255, np.uint8)
            _, mask_wp = warper.warp(mask, K, self.cameras[i].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

            corners.append(corner)
            sizes.append(size)
            images_warped.append(img_wp)
            masks_warped.append(mask_wp)

        images_warped_f = [img.astype(np.float32) for img in images_warped]
        self.exposure_comp.feed(corners, images_warped, masks_warped)
        for i in range(len(images_warped)):
            self.exposure_comp.apply(i, corners[i], images_warped[i], masks_warped[i])

        self.seam_finder.find(images_warped_f, corners, masks_warped)
        self.blender.prepare(corners, sizes)

        for idx in range(len(self.imgs)):
            img = self.imgs[idx]
            if abs(compose_scale - 1) > 1e-1:
                img = cv2.resize(img, None, fx=compose_scale, fy=compose_scale, interpolation=cv2.INTER_LINEAR)

            K = self.cameras[idx].K().astype(np.float32) * compose_work_aspect
            K[2, 2] = 1.0

            _, img_warped = warper.warp(img, K, self.cameras[idx].R, self.interp_flags, cv2.BORDER_REFLECT)
            mask = np.full(img.shape[:2], 255, np.uint8)
            _, mask_warped = warper.warp(mask, K, self.cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

            self.exposure_comp.apply(idx, corners[idx], img_warped, mask_warped)

            dilated = cv2.dilate(masks_warped[idx], None)
            seam_mask = cv2.resize(dilated, (mask_warped.shape[1], mask_warped.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_warped = cv2.bitwise_and(seam_mask, mask_warped)

            self.blender.feed(img_warped.astype(np.int16), mask_warped, corners[idx])

        result, _ = self.blender.blend(None, None)
        result = cv2.convertScaleAbs(result)
        return self.Status.OK, result