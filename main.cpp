#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>       // cho fs::create_directory
#include <ctime>
#include <iomanip>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;
using namespace cv::detail;

// --------- Configs ---------
const int PYR_LEVELS = 4;
const float PYR_SCALE = 1.2f;
const int BRIEF_SIZE_BITS = 256; // 256 bits -> 32 bytes
const int MAX_PANORAMA_DIM = 12000; // giới hạn an toàn (tùy máy mà chỉnh)
const int MIN_GOOD_MATCHES = 8;    // yêu cầu tối thiểu để tính homography

// Resize large images early to save memory (adjust maxDimension as needed)
Mat safeResizeMaxDim(const Mat &img, int maxDim = 2500) {
    int w = img.cols, h = img.rows;
    int maxwh = max(w, h);
    if (maxwh <= maxDim) return img.clone();
    double scale = double(maxDim) / double(maxwh);
    Mat out;
    resize(img, out, Size(), scale, scale, INTER_AREA);
    return out;
}

vector<Mat> buildPyramid(const Mat& img, int nlevels = PYR_LEVELS, float scaleFactor = PYR_SCALE) {
    vector<Mat> pyr(nlevels);
    pyr[0] = img.clone();
    for (int i = 1; i < nlevels; ++i) {
        float scale = pow(scaleFactor, i);
        resize(img, pyr[i], Size(), 1.0 / scale, 1.0 / scale, INTER_LINEAR);
    }
    return pyr;
}

float computeOrientation(const Mat& img, Point2f pt) {
    const int half = 15;
    float m01 = 0, m10 = 0;
    for (int u = -half; u <= half; ++u) {
        for (int v = -half; v <= half; ++v) {
            int x = cvRound(pt.x + u);
            int y = cvRound(pt.y + v);
            if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) continue;
            uchar I = img.at<uchar>(y, x);
            m10 += u * I;
            m01 += v * I;
        }
    }
    return atan2f(m01, m10);
}

Mat computeBRIEF(const Mat& img, Point2f pt, float angle, int &validPairs) {
    static vector<Point> pattern;
    if (pattern.empty()) {
        RNG rng(0x12345);
        pattern.reserve(BRIEF_SIZE_BITS * 2);
        for (int i = 0; i < BRIEF_SIZE_BITS; ++i) {
            pattern.emplace_back(rng.uniform(-15, 16), rng.uniform(-15, 16));
            pattern.emplace_back(rng.uniform(-15, 16), rng.uniform(-15, 16));
        }
    }

    Mat desc(1, BRIEF_SIZE_BITS/8, CV_8U, Scalar(0));
    float c = cos(angle), s = sin(angle);
    validPairs = 0;

    for (int i = 0; i < BRIEF_SIZE_BITS; ++i) {
        Point2f p1 = pattern[2*i];
        Point2f p2 = pattern[2*i+1];

        Point2f r1(c*p1.x - s*p1.y, s*p1.x + c*p1.y);
        Point2f r2(c*p2.x - s*p2.y, s*p2.x + c*p2.y);

        int x1 = cvRound(pt.x + r1.x), y1 = cvRound(pt.y + r1.y);
        int x2 = cvRound(pt.x + r2.x), y2 = cvRound(pt.y + r2.y);

        if (x1 < 0 || x1 >= img.cols || y1 < 0 || y1 >= img.rows ||
            x2 < 0 || x2 >= img.cols || y2 < 0 || y2 >= img.rows) continue;

        ++validPairs;
        if (img.at<uchar>(y1, x1) < img.at<uchar>(y2, x2))
            desc.at<uchar>(i/8) |= (1 << (i%8));
    }
    return desc;
}

// Revised extractFeatures: compute descriptors into vector<Mat>, skip invalid descriptors,
// scale keypoint coordinates correctly (from pyramid coords -> original image coords)
void extractFeatures(const Mat& gray, vector<KeyPoint>& kp, Mat& descriptors)
{
    kp.clear();
    descriptors.release();

    auto pyr = buildPyramid(gray, PYR_LEVELS, PYR_SCALE);

    vector<vector<KeyPoint>> temp_kpts(PYR_LEVELS);
    for (int l = 0; l < PYR_LEVELS; ++l) {
        FAST(pyr[l], temp_kpts[l], 20, true);
    }

    vector<KeyPoint> kp_out;
    vector<Mat> descs;

    for (int lvl = 0; lvl < PYR_LEVELS; ++lvl) {
        float scale = pow(PYR_SCALE, lvl);          // scale >=1 : pyr[lvl] is smaller by factor scale
        for (auto &k0 : temp_kpts[lvl]) {
            // k0.pt is coordinate on pyr[lvl]
            Point2f pt_pyr = k0.pt;

            // compute orientation & BRIEF on pyramid image
            float angle = computeOrientation(pyr[lvl], pt_pyr);
            int validPairs = 0;
            Mat desc = computeBRIEF(pyr[lvl], pt_pyr, angle, validPairs);

            // require enough valid pairs to accept descriptor
            if (validPairs < BRIEF_SIZE_BITS/8) {
                // too few valid comparisons -> skip keypoint
                continue;
            }

            // create keypoint in original image coordinates (scale up)
            KeyPoint k;
            k.pt.x = pt_pyr.x * scale;
            k.pt.y = pt_pyr.y * scale;
            k.size = 31.0f * scale;
            k.octave = lvl;
            k.angle = angle * 180.0f / CV_PI;

            kp_out.push_back(k);
            descs.push_back(desc);
        }
    }

    // assemble descriptors Mat
    if (!descs.empty()) {
        descriptors.create((int)descs.size(), BRIEF_SIZE_BITS/8, CV_8U);
        for (size_t i = 0; i < descs.size(); ++i) {
            descs[i].copyTo(descriptors.row((int)i));
        }
    }
    kp = std::move(kp_out);
}

// Best-of-2-NN with cross-check (unchanged other than safety)
struct MyBestOf2NearestMatcher {
    void match(const Mat& desc1, const Mat& desc2,
            const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2,
            vector<DMatch>& good)
    {
        good.clear();
        if (desc1.empty() || desc2.empty()) return;

        BFMatcher matcher(NORM_HAMMING);
        vector<vector<DMatch>> knn12, knn21;

        matcher.knnMatch(desc1, desc2, knn12, 2);
        matcher.knnMatch(desc2, desc1, knn21, 2);   // cross-check

        for (int i = 0; i < (int)knn12.size(); ++i) {
            if (knn12[i].size() < 2) continue;
            const DMatch& m1 = knn12[i][0];
            const DMatch& m2 = knn12[i][1];

            if (m1.distance >= 0.75f * m2.distance) continue;
            if (m1.distance > 90) continue;

            // cross-check
            if (m1.trainIdx >= (int)knn21.size()) continue;
            if (knn21[m1.trainIdx].empty()) continue;
            if (knn21[m1.trainIdx][0].trainIdx != i) continue;

            good.push_back(m1);
        }

        sort(good.begin(), good.end(),
            [](const DMatch& a, const DMatch& b) { return a.distance < b.distance; });
    }
};

// Helper: check matrix finite
bool matIsFinite(const Mat& M) {
    for (int r = 0; r < M.rows; ++r) {
        for (int c = 0; c < M.cols; ++c) {
            double v = M.at<double>(r,c);
            if (!isfinite(v)) return false;
            if (fabs(v) > 1e9) return false; // guardrail
        }
    }
    return true;
}

// Helper: return bounding rect of non-zero region (or empty rect)
Rect nonZeroBoundingRect(const Mat &img) {
    Mat gray, nz;
    if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);
    else gray = img;
    threshold(gray, nz, 1, 255, THRESH_BINARY);
    vector<vector<Point>> contours; findContours(nz, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return Rect();
    Rect bb = boundingRect(contours[0]);
    for (size_t i=1;i<contours.size();++i) bb |= boundingRect(contours[i]);
    return bb;
}

int main() {
    // Load full-resolution originals
    Mat img1_color_full = imread("img1.jpg");
    Mat img2_color_full = imread("img2.jpg");
    if (img1_color_full.empty() || img2_color_full.empty()) {
        cout << "Khong tim thay img1.jpg hoac img2.jpg";
        return -1;
    }

    if (!fs::exists("output")) fs::create_directory("output");

    // 1) Input Images -> Resize (medium resolution)
    const int INPUT_MAX_DIM = 2500; // medium resolution
    Mat img1_med = safeResizeMaxDim(img1_color_full, INPUT_MAX_DIM);
    Mat img2_med = safeResizeMaxDim(img2_color_full, INPUT_MAX_DIM);
    imwrite("output/01_input_resize_medium_img1.jpg", img1_med);
    imwrite("output/01_input_resize_medium_img2.jpg", img2_med);

    // 2) Find features (on medium res)
    Mat gray1_med, gray2_med;
    cvtColor(img1_med, gray1_med, COLOR_BGR2GRAY);
    cvtColor(img2_med, gray2_med, COLOR_BGR2GRAY);
    vector<KeyPoint> kp1_med, kp2_med;
    Mat desc1_med, desc2_med;
    cout << "Find features (medium)..." << endl;
    extractFeatures(gray1_med, kp1_med, desc1_med);
    extractFeatures(gray2_med, kp2_med, desc2_med);

    // draw keypoints (for slide)
    Mat feat_vis1, feat_vis2;
    drawKeypoints(img1_med, kp1_med, feat_vis1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(img2_med, kp2_med, feat_vis2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite("output/02_features_img1_med.jpg", feat_vis1);
    imwrite("output/02_features_img2_med.jpg", feat_vis2);

    // 3) Match features
    vector<DMatch> good_med;
    MyBestOf2NearestMatcher().match(desc1_med, desc2_med, kp1_med, kp2_med, good_med);
    cout << "Found matches (medium): " << good_med.size() << endl;
    if ((int)good_med.size() < MIN_GOOD_MATCHES) {
        cout << "Too few matches to compute reliable homography." << endl;
        return -1;
    }

    // draw matches for slide
    int show = min(500, (int)good_med.size());
    vector<DMatch> show_matches(good_med.begin(), good_med.begin() + show);
    Mat match_vis;
    drawMatches(img1_med, kp1_med, img2_med, kp2_med, show_matches, match_vis);
    imwrite("output/03_matches_medium.jpg", match_vis);

    // 4) Prepare correspondences for homography and compute H_med with RANSAC and get mask
    vector<Point2f> pts1_med, pts2_med;
    for (const auto &m : good_med) {
        if (m.queryIdx < (int)kp1_med.size() && m.trainIdx < (int)kp2_med.size()) {
            pts1_med.push_back(kp1_med[m.queryIdx].pt);
            pts2_med.push_back(kp2_med[m.trainIdx].pt);
        }
    }
    Mat mask_inliers;
    Mat H_med = findHomography(pts2_med, pts1_med, RANSAC, 3.0, mask_inliers);
    if (H_med.empty()) {
        cout << "Khong the tinh Homography (medium)" << endl;
        return -1;
    }
    // print inliers count for debug
    int inliers = 0;
    for (int i = 0; i < mask_inliers.rows; ++i) if (mask_inliers.at<uchar>(i)) ++inliers;
    cout << "H_med computed. Inliers: " << inliers << " / " << mask_inliers.rows << endl;
    H_med.convertTo(H_med, CV_64F);

    // Compute scaling matrices to map H_med to low-res coordinates
    const int LOWRES_MAX_DIM = 800; // low resolution for fast computations
    Mat img1_low, img2_low;
    resize(img1_med, img1_low, Size(), double(LOWRES_MAX_DIM)/double(max(img1_med.cols, img1_med.rows)), double(LOWRES_MAX_DIM)/double(max(img1_med.cols, img1_med.rows)), INTER_AREA);
    resize(img2_med, img2_low, Size(), double(LOWRES_MAX_DIM)/double(max(img2_med.cols, img2_med.rows)), double(LOWRES_MAX_DIM)/double(max(img2_med.cols, img2_med.rows)), INTER_AREA);
    imwrite("output/05_input_resize_low_img1.jpg", img1_low);
    imwrite("output/05_input_resize_low_img2.jpg", img2_low);

    double s1 = double(img1_low.cols) / double(img1_med.cols);
    double s2 = double(img2_low.cols) / double(img2_med.cols);
    Mat S1 = Mat::eye(3,3,CV_64F); S1.at<double>(0,0) = S1.at<double>(1,1) = s1;
    Mat S2 = Mat::eye(3,3,CV_64F); S2.at<double>(0,0) = S2.at<double>(1,1) = s2;
    Mat H_low = S1 * H_med * S2.inv();  // CORRECT formula

    // Warp low-res into a canvas using the robust bbox-from-warp method
    vector<Point2f> corners2_low = { Point2f(0,0), Point2f((float)img2_low.cols,0), Point2f((float)img2_low.cols,(float)img2_low.rows), Point2f(0,(float)img2_low.rows) };
    vector<Point2f> corners2_low_t; perspectiveTransform(corners2_low, corners2_low_t, H_low);
    float min_x_low = 0, min_y_low = 0, max_x_low = (float)img1_low.cols, max_y_low = (float)img1_low.rows;
    for (auto &p : corners2_low_t) { min_x_low = min(min_x_low, p.x); min_y_low = min(min_y_low, p.y); max_x_low = max(max_x_low, p.x); max_y_low = max(max_y_low, p.y); }

    int approx_w = (int)ceil(max_x_low - min_x_low);
    int approx_h = (int)ceil(max_y_low - min_y_low);
    approx_w = max(approx_w, img1_low.cols);
    approx_h = max(approx_h, img1_low.rows);

    Mat H1_tmp = Mat::eye(3,3,CV_64F);
    Mat tmp_w1, tmp_w2;
    warpPerspective(img1_low, tmp_w1, H1_tmp, Size(approx_w, approx_h));
    warpPerspective(img2_low, tmp_w2, H_low, Size(approx_w, approx_h));

    Rect bb1 = nonZeroBoundingRect(tmp_w1);
    Rect bb2 = nonZeroBoundingRect(tmp_w2);

    if (bb1.width == 0 || bb1.height == 0) bb1 = Rect(0,0,img1_low.cols,img1_low.rows);
    if (bb2.width == 0 || bb2.height == 0) bb2 = Rect(0,0,img2_low.cols,img2_low.rows);

    int final_min_x = min(0, min(bb1.x, bb2.x));
    int final_min_y = min(0, min(bb1.y, bb2.y));
    int final_max_x = max(bb1.x + bb1.width, bb2.x + bb2.width);
    int final_max_y = max(bb1.y + bb1.height, bb2.y + bb2.height);

    int shift_x_low = (final_min_x < 0) ? -final_min_x : 0;
    int shift_y_low = (final_min_y < 0) ? -final_min_y : 0;
    int pano_w_low = final_max_x - final_min_x;
    int pano_h_low = final_max_y - final_min_y;

    if (pano_w_low <=0 || pano_h_low<=0) { cout<<"Low-res pano invalid"; return -1; }

    Mat Tlow = Mat::eye(3,3,CV_64F); Tlow.at<double>(0,2)=shift_x_low; Tlow.at<double>(1,2)=shift_y_low;
    Mat H2_low_shifted = Tlow * H_low;

    Mat warp1_low, warp2_low;
    Mat H1_low = Mat::eye(3,3,CV_64F); H1_low.at<double>(0,2)=shift_x_low; H1_low.at<double>(1,2)=shift_y_low;
    warpPerspective(img1_low, warp1_low, H1_low, Size(pano_w_low, pano_h_low));
    warpPerspective(img2_low, warp2_low, H2_low_shifted, Size(pano_w_low, pano_h_low));
    imwrite("output/06_warp_low_img1.jpg", warp1_low);
    imwrite("output/06_warp_low_img2.jpg", warp2_low);

    // 7) Exposure compensator (low-res)
    Ptr<ExposureCompensator> expCompLow = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
    BlocksGainCompensator *bgcLow = dynamic_cast<BlocksGainCompensator*>(expCompLow.get());
    if (bgcLow) bgcLow->setBlockSize(16,16);
    Point corner_img1_low(shift_x_low, shift_y_low);
    Rect bbox2_low = nonZeroBoundingRect(warp2_low);
    Point corner_img2_low = (bbox2_low.width>0 && bbox2_low.height>0) ? bbox2_low.tl() : Point(0,0);
    vector<Point> corners_low = { corner_img1_low, corner_img2_low };

    vector<UMat> imgs_low_umat(2); imgs_low_umat[0]=warp1_low.getUMat(ACCESS_RW); imgs_low_umat[1]=warp2_low.getUMat(ACCESS_RW);
    vector<UMat> masks_low_umat(2);
    Mat m1_low = Mat::zeros(warp1_low.size(), CV_8U); m1_low.setTo(255);
    Mat m2_low;
    Mat nz_low;
    if (bbox2_low.width > 0 && bbox2_low.height > 0) {
        cvtColor(warp2_low, nz_low, COLOR_BGR2GRAY);
        threshold(nz_low, nz_low, 1, 255, THRESH_BINARY);
        m2_low = nz_low.clone();
    } else {
        m2_low = Mat::zeros(warp2_low.size(), CV_8U);
    }
    masks_low_umat[0]=m1_low.getUMat(ACCESS_RW); masks_low_umat[1]=m2_low.getUMat(ACCESS_RW);

    expCompLow->feed(corners_low, imgs_low_umat, masks_low_umat);
    Mat warp1_low_copy = warp1_low.clone(), warp2_low_copy = warp2_low.clone();
    expCompLow->apply(0, Point(0,0), warp1_low_copy, m1_low);
    expCompLow->apply(1, Point(0,0), warp2_low_copy, m2_low);
    Mat diff_low; absdiff(warp2_low, warp2_low_copy, diff_low);
    imwrite("output/07_exposure_before_img2_low.jpg", warp2_low);
    imwrite("output/07_exposure_after_img2_low.jpg", warp2_low_copy);
    imwrite("output/07_exposure_diff_img2_low.jpg", diff_low);

    // 8) Seam finding (low-res)
    Mat mask1_low = Mat::zeros(warp1_low.size(), CV_8U); mask1_low.setTo(255);
    Mat mask2_low = Mat::zeros(warp2_low.size(), CV_8U);
    if (bbox2_low.width > 0 && bbox2_low.height > 0) mask2_low = nz_low.clone();
    else mask2_low.setTo(0);

    masks_low_umat[0] = mask1_low.getUMat(ACCESS_RW);
    masks_low_umat[1] = mask2_low.getUMat(ACCESS_RW);

    Mat tmp32;
    warp1_low.convertTo(tmp32, CV_32F); imgs_low_umat[0] = tmp32.getUMat(ACCESS_RW);
    warp2_low.convertTo(tmp32, CV_32F); imgs_low_umat[1] = tmp32.getUMat(ACCESS_RW);

    Ptr<GraphCutSeamFinder> seamFinder = makePtr<GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    Mat mask1_low_res, mask2_low_res;
    Mat overlap_tmp;
    bitwise_and(mask1_low, mask2_low, overlap_tmp);
    if (countNonZero(overlap_tmp) > 0) {
        seamFinder->find(imgs_low_umat, corners_low, masks_low_umat);
        masks_low_umat[0].getMat(ACCESS_READ).copyTo(mask1_low_res);
        masks_low_umat[1].getMat(ACCESS_READ).copyTo(mask2_low_res);
    } else {
        mask1_low_res = mask1_low.clone();
        mask2_low_res = mask2_low.clone();
    }
    imwrite("output/08_seam_mask_low_img1.png", mask1_low_res);
    imwrite("output/08_seam_mask_low_img2.png", mask2_low_res);

    // 9) Compute full-resolution H (map full img2 -> full img1)
    double scale1_full_to_med = double(img1_color_full.cols) / double(img1_med.cols);
    double scale2_full_to_med = double(img2_color_full.cols) / double(img2_med.cols);
    Mat S1full = Mat::eye(3,3,CV_64F); S1full.at<double>(0,0)=S1full.at<double>(1,1)=scale1_full_to_med;
    Mat S2full = Mat::eye(3,3,CV_64F); S2full.at<double>(0,0)=S2full.at<double>(1,1)=scale2_full_to_med;
    Mat H_full = S1full * H_med * S2full.inv();

    // Tính bounding box full panorama
    vector<Point2f> corners2_full = {
        Point2f(0,0),
        Point2f((float)img2_color_full.cols, 0),
        Point2f((float)img2_color_full.cols, (float)img2_color_full.rows),
        Point2f(0, (float)img2_color_full.rows)
    };
    vector<Point2f> corners2_warped; perspectiveTransform(corners2_full, corners2_warped, H_full);

    float min_x = 0, min_y = 0;
    float max_x = img1_color_full.cols;
    float max_y = img1_color_full.rows;
    for (const auto& p : corners2_warped) {
        min_x = min(min_x, p.x);
        min_y = min(min_y, p.y);
        max_x = max(max_x, p.x);
        max_y = max(max_y, p.y);
    }

    int offset_x = (min_x < 0) ? (int)ceil(-min_x) : 0;
    int offset_y = (min_y < 0) ? (int)ceil(-min_y) : 0;
    int pano_width  = (int)ceil(max_x - min_x);
    int pano_height = (int)ceil(max_y - min_y);

    Mat T_full = Mat::eye(3,3,CV_64F);
    T_full.at<double>(0,2) = offset_x;
    T_full.at<double>(1,2) = offset_y;

    Mat H1_full_final = T_full.clone();
    Mat H2_full_final = T_full * H_full;

    // Use pano size to compute scale between low and full pano
    if (pano_w_low == 0 || pano_h_low == 0) {
        cout << "Invalid low-res pano dims for scaling." << endl;
        return -1;
    }
    double scale_pano_x = double(pano_width)  / double(pano_w_low);
    double scale_pano_y = double(pano_height) / double(pano_h_low);

    int offset_x_low = (int)round(offset_x / max(1e-9, scale_pano_x));
    int offset_y_low = (int)round(offset_y / max(1e-9, scale_pano_y));

    Mat T_low_final = Mat::eye(3,3,CV_64F);
    T_low_final.at<double>(0,2) = offset_x_low;
    T_low_final.at<double>(1,2) = offset_y_low;

    Mat H1_low_final = T_low_final.clone();
    Mat H2_low_final = T_low_final * H_low;

    Mat warp1_low_new, warp2_low_new;
    warpPerspective(img1_low, warp1_low_new, H1_low_final, Size(pano_width, pano_height));
    warpPerspective(img2_low, warp2_low_new, H2_low_final, Size(pano_width, pano_height));

    // For full-level bb detection, use actual full warps (we already computed H1_full_final/H2_full_final)
    Mat warp1_full, warp2_full;
    
    warpPerspective(img1_color_full, warp1_full, H1_full_final, Size(pano_width, pano_height));
    warpPerspective(img2_color_full, warp2_full, H2_full_final, Size(pano_width, pano_height));
    imwrite("output/10_warp_full_img1.jpg", warp1_full);
    imwrite("output/10_warp_full_img2.jpg", warp2_full);

    Rect bb1_full = nonZeroBoundingRect(warp1_full);
    Rect bb2_full = nonZeroBoundingRect(warp2_full);
    if (bb1_full.width==0 || bb1_full.height==0) bb1_full = Rect(0,0,img1_color_full.cols,img1_color_full.rows);
    if (bb2_full.width==0 || bb2_full.height==0) bb2_full = Rect(0,0,img2_color_full.cols,img2_color_full.rows);

    int final_min_xf = min(0, min(bb1_full.x, bb2_full.x));
    int final_min_yf = min(0, min(bb1_full.y, bb2_full.y));
    int final_max_xf = max(bb1_full.x + bb1_full.width, bb2_full.x + bb2_full.width);
    int final_max_yf = max(bb1_full.y + bb1_full.height, bb2_full.y + bb2_full.height);

    int shift_x_full = (final_min_xf < 0) ? -final_min_xf : 0;
    int shift_y_full = (final_min_yf < 0) ? -final_min_yf : 0;
    int pano_w_full = final_max_xf - final_min_xf;
    int pano_h_full = final_max_yf - final_min_yf;
    if (pano_w_full <=0 || pano_h_full<=0) { cout<<"Full pano invalid"; return -1; }

    // Create NZ masks full
    Mat gray_w1_full, gray_w2_full, mask1_full_nz, mask2_full_nz;
    cvtColor(warp1_full, gray_w1_full, COLOR_BGR2GRAY);
    cvtColor(warp2_full, gray_w2_full, COLOR_BGR2GRAY);
    threshold(gray_w1_full, mask1_full_nz, 1, 255, THRESH_BINARY);
    threshold(gray_w2_full, mask2_full_nz, 1, 255, THRESH_BINARY);

    // Upsample low-res seam masks to full pano size and intersect with nz masks
    Mat up_mask1_full, up_mask2_full;
    if (mask1_low_res.empty()) mask1_low_res = Mat::zeros(pano_h_low, pano_w_low, CV_8U);
    if (mask2_low_res.empty()) mask2_low_res = Mat::zeros(pano_h_low, pano_w_low, CV_8U);
    resize(mask1_low_res, up_mask1_full, Size(pano_w_full, pano_h_full), 0, 0, INTER_NEAREST);
    resize(mask2_low_res, up_mask2_full, Size(pano_w_full, pano_h_full), 0, 0, INTER_NEAREST);

    Mat mask1_full = (up_mask1_full & mask1_full_nz);
    Mat mask2_full = (up_mask2_full & mask2_full_nz);

    imwrite("debug_mask1_full.png", mask1_full);
    imwrite("debug_mask2_full.png", mask2_full);
    Mat overlap_full = mask1_full & mask2_full;
    cout << "Overlap pixels full: " << countNonZero(overlap_full) << endl;
    
    if (countNonZero(mask1_full) == 0) mask1_full = mask1_full_nz.clone();
    if (countNonZero(mask2_full) == 0) mask2_full = mask2_full_nz.clone();

    imwrite("output/09_seam_mask_full_img1.png", mask1_full);
    imwrite("output/09_seam_mask_full_img2.png", mask2_full);

    // 11) Exposure compensation full: use local feed
    // Ptr<ExposureCompensator> expCompFull = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
    // BlocksGainCompensator *bgcFull = dynamic_cast<BlocksGainCompensator*>(expCompFull.get()); if (bgcFull) bgcFull->setBlockSize(32,32);
    // vector<UMat> imgs_full_umat(2); imgs_full_umat[0]=warp1_full.getUMat(ACCESS_RW); imgs_full_umat[1]=warp2_full.getUMat(ACCESS_RW);
    // vector<UMat> masks_full_umat(2); masks_full_umat[0]=mask1_full.getUMat(ACCESS_RW); masks_full_umat[1]=mask2_full.getUMat(ACCESS_RW);

    // vector<Point> corners_full_vec;
    // Point corner_img1_full(shift_x_full, shift_y_full); // FIXED (was shift_x_full, shift_x_full)
    // // compute bbox of warp2_full content
    // vector<vector<Point>> contours_full2; findContours(mask2_full_nz, contours_full2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // Point corner_img2_full_pt(0,0);
    // if (!contours_full2.empty()) {
    //     Rect bbox2_full = boundingRect(contours_full2[0]);
    //     for (size_t i=1;i<contours_full2.size();++i) bbox2_full |= boundingRect(contours_full2[i]);
    //     corner_img2_full_pt = bbox2_full.tl();
    // }
    // corners_full_vec.push_back(corner_img1_full);
    // corners_full_vec.push_back(corner_img2_full_pt);

    // expCompFull->feed(corners_full_vec, imgs_full_umat, masks_full_umat);
    // Mat warp1_full_copy=warp1_full.clone(), warp2_full_copy=warp2_full.clone();
    // expCompFull->apply(0, Point(0,0), warp1_full_copy, mask1_full);
    // expCompFull->apply(1, Point(0,0), warp2_full_copy, mask2_full);
    // imwrite("output/11_compensated_full_img1.jpg", warp1_full_copy);
    // imwrite("output/11_compensated_full_img2.jpg", warp2_full_copy);

    // Log canvas size
    cout << "warp1_full size: " << warp1_full.cols << "x" << warp1_full.rows << endl;
    cout << "warp2_full size: " << warp2_full.cols << "x" << warp2_full.rows << endl;

    // Log mask info
    cout << "mask1_full nonzero pixels: " << countNonZero(mask1_full) << endl;
    cout << "mask2_full nonzero pixels: " << countNonZero(mask2_full) << endl;

    cout << "Overlap pixels full: " << countNonZero(overlap_full) << endl;

    // Log corners
   // corner1_full = top-left nonzero của warp1_full
    Mat nz1_full; cvtColor(warp1_full, nz1_full, COLOR_BGR2GRAY); threshold(nz1_full, nz1_full, 1, 255, THRESH_BINARY);
    bb1_full = boundingRect(nz1_full);
    Point corner1 = bb1_full.tl();

    // corner2_full = top-left nonzero của warp2_full
    Mat nz2_full; cvtColor(warp2_full, nz2_full, COLOR_BGR2GRAY); threshold(nz2_full, nz2_full, 1, 255, THRESH_BINARY);
    bb2_full = boundingRect(nz2_full);
    Point corner2 = bb2_full.tl();

    vector<Point> corners_full_vec = {corner1, corner2};
    cout << "Corners fed to compensator: img1=" << corner1 << ", img2=" << corner2 << endl;

    // Prepare compensator
    Ptr<ExposureCompensator> expCompFull = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
    BlocksGainCompensator *bgcFull = dynamic_cast<BlocksGainCompensator*>(expCompFull.get());
    if(bgcFull) {
        bgcFull->setBlockSize(32,32);
    }

    // Feed images + masks
    vector<UMat> imgs_full_umat(2);
    imgs_full_umat[0] = warp1_full.getUMat(ACCESS_RW);
    imgs_full_umat[1] = warp2_full.getUMat(ACCESS_RW);

    vector<UMat> masks_full_umat(2);
    masks_full_umat[0] = mask1_full.getUMat(ACCESS_RW);
    masks_full_umat[1] = mask2_full.getUMat(ACCESS_RW);

    expCompFull->feed(corners_full_vec, imgs_full_umat, masks_full_umat);

    // Apply compensator
    Mat warp1_full_copy = warp1_full.clone();
    Mat warp2_full_copy = warp2_full.clone();
    expCompFull->apply(0, Point(0,0), warp1_full_copy, mask1_full);
    expCompFull->apply(1, Point(0,0), warp2_full_copy, mask2_full);

    // Debug diff
    Mat diff1, diff2;
    absdiff(warp1_full, warp1_full_copy, diff1);
    absdiff(warp2_full, warp2_full_copy, diff2);
    cout << "diff1 non-zero pixels: " << countNonZero(diff1.reshape(1)) << endl;
    cout << "diff2 non-zero pixels: " << countNonZero(diff2.reshape(1)) << endl;

    // Save outputs
    imwrite("debug_full_exposure_before_img1.jpg", warp1_full);
    imwrite("debug_full_exposure_after_img1.jpg", warp1_full_copy);
    imwrite("debug_full_exposure_before_img2.jpg", warp2_full);
    imwrite("debug_full_exposure_after_img2.jpg", warp2_full_copy);
    imwrite("debug_full_exposure_diff1.jpg", diff1);
    imwrite("debug_full_exposure_diff2.jpg", diff2);
    // 12) Blend images (use MultiBandBlender with limited bands)
    Ptr<MultiBandBlender> mb = makePtr<MultiBandBlender>(false);
    int maxDim = max(pano_w_full, pano_h_full);
    int numBands = 1; if (maxDim>4000) numBands=2; if (maxDim>8000) numBands=3;
    mb->setNumBands(numBands);
    mb->prepare(Rect(0,0,pano_w_full,pano_h_full));

    Mat w1_s16, w2_s16;
    warp1_full_copy.convertTo(w1_s16, CV_16S);
    warp2_full_copy.convertTo(w2_s16, CV_16S);

    mb->feed(w1_s16, mask1_full, Point(0,0));
    mb->feed(w2_s16, mask2_full, Point(0,0));

    Mat result_s16, result_mask;
    mb->blend(result_s16, result_mask);
    Mat final; result_s16.convertTo(final, CV_8U);
    imwrite("output/12_blended_full.jpg", final);

    cout << "Exported all intermediate images to ./output (steps 01..12)." << endl;
    return 0;
}
