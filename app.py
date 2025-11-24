import os
import uuid
import io
from typing import List
from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np

from org_stitcher import crop_black_borders
import custom_stitcher as cs

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(APP_ROOT, "web_uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

# Serve templates/static from the `web/` folder we created
app = Flask(__name__, template_folder=os.path.join(APP_ROOT, 'web', 'templates'), static_folder=os.path.join(APP_ROOT, 'web', 'static'))


def save_image_file(folder: str, file_storage) -> str:
    filename = file_storage.filename
    path = os.path.join(folder, filename)
    file_storage.save(path)
    return filename


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No files'}), 400

    session_id = uuid.uuid4().hex
    session_folder = os.path.join(UPLOAD_ROOT, session_id)
    os.makedirs(session_folder, exist_ok=True)

    saved = []
    for f in files:
        # sanitize filename simply
        filename = save_image_file(session_folder, f)
        saved.append(filename)

    return jsonify({'session': session_id, 'files': saved})


@app.route('/uploads/<session>/<filename>')
def serve_upload(session, filename):
    path = os.path.join(UPLOAD_ROOT, session, filename)
    if not os.path.exists(path):
        return 'Not found', 404
    return send_file(path)


@app.route('/keypoints/<session>/<int:idx>')
def keypoints(session, idx):
    folder = os.path.join(UPLOAD_ROOT, session)
    files = sorted(os.listdir(folder))
    if idx < 0 or idx >= len(files):
        return 'Index out of range', 400
    path = os.path.join(folder, files[idx])
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, desc = cs.detect_and_describe(gray)
    out = cv2.drawKeypoints(img, kps, None, color=(0,255,0))
    return _image_response(out)


@app.route('/matches/<session>/<int:i>/<int:j>')
def matches(session, i, j):
    folder = os.path.join(UPLOAD_ROOT, session)
    files = sorted(os.listdir(folder))
    if any(x<0 or x>=len(files) for x in (i,j)):
        return 'Index out of range', 400
    A = cv2.imread(os.path.join(folder, files[i]))
    B = cv2.imread(os.path.join(folder, files[j]))
    grayA = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    kpsA, descA = cs.detect_and_describe(grayA)
    kpsB, descB = cs.detect_and_describe(grayB)
    matches = cs.match_features(descA, descB)
    # convert to list of DMatch for drawMatches
    img_matches = cv2.drawMatches(A, kpsA, B, kpsB, matches, None, flags=2)
    return _image_response(img_matches)


@app.route('/warp/<session>/<int:i>/<int:j>')
def warp(session, i, j):
    folder = os.path.join(UPLOAD_ROOT, session)
    files = sorted(os.listdir(folder))
    if any(x<0 or x>=len(files) for x in (i,j)):
        return 'Index out of range', 400
    A = cv2.imread(os.path.join(folder, files[i]))
    B = cv2.imread(os.path.join(folder, files[j]))
    grayA = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    kpsA, descA = cs.detect_and_describe(grayA)
    kpsB, descB = cs.detect_and_describe(grayB)
    matches = cs.match_features(descA, descB)
    H = cs.estimate_homography(kpsA, kpsB, matches)
    if H is None:
        return jsonify({'error': 'Homography estimation failed'}), 400

    warped = _warp_B_to_A(A, B, H)
    return _image_response(warped)


@app.route('/stitch_custom/<session>')
def stitch_custom(session):
    folder = os.path.join(UPLOAD_ROOT, session)
    files = sorted(os.listdir(folder))
    imgs = [cv2.imread(os.path.join(folder, f)) for f in files]
    # call custom stitch
    try:
        pano = cs.stitch(imgs)
    except Exception as e:
        return jsonify({'error': f'stitch exception: {e}'}), 500

    # cs.stitch may return tuple or image
    if isinstance(pano, tuple):
        # if (status, img)
        if len(pano) >= 2:
            pano_img = pano[1]
        else:
            return jsonify({'error': 'unexpected return from stitch'}), 500
    else:
        pano_img = pano

    if pano_img is None:
        return jsonify({'error': 'stitch failed'}), 500

    # save temporarily and also return cropped
    cropped = crop_black_borders(pano_img, margin=8)

    return _image_response(cropped)


@app.route('/stitch_fast/<session>')
def stitch_fast(session):
    """Fast but good-quality stitch: run OpenCV C++ stitcher on downscaled images
    to keep runtime bounded. We downscale so the longest side <= 1600 px.
    """
    folder = os.path.join(UPLOAD_ROOT, session)
    if not os.path.isdir(folder):
        return 'Session not found', 404
    files = sorted(os.listdir(folder))
    if not files:
        return 'No files', 400

    imgs = [cv2.imread(os.path.join(folder, f)) for f in files]
    # downscale for speed while keeping aspect
    def rescale(img, max_side=1600):
        h,w = img.shape[:2]
        s = max(h,w)
        if s <= max_side:
            return img
        scale = max_side / float(s)
        return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    small = [rescale(im, max_side=1600) for im in imgs]

    try:
        stitcher = None
        # prefer cv2.Stitcher_create where available
        if hasattr(cv2, 'Stitcher_create'):
            stitcher = cv2.Stitcher_create()
        else:
            stitcher = cv2.createStitcher(False)
        status, pano = stitcher.stitch(small)
    except Exception as e:
        return jsonify({'error': f'stitcher exception: {e}'}), 500

    if status != cv2.Stitcher_OK and status != 0:
        return jsonify({'error': f'stitch failed, status={status}'}), 500

    if pano is None:
        return jsonify({'error': 'stitch produced no image'}), 500

    try:
        cropped = crop_black_borders(pano, margin=8)
    except Exception:
        cropped = pano

    return _image_response(cropped)



def _warp_B_to_A(A, B, H, apply_exposure=True):
    if apply_exposure:
        gain = compute_exposure_gain(A, B, H)
        B = apply_gain(B, gain)
    hA, wA = A.shape[:2]
    hB, wB = B.shape[:2]
    cornersB = np.float32([[0,0],[wB,0],[wB,hB],[0,hB]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(cornersB, H)
    all_corners = np.concatenate((warped_corners, np.float32([[0,0],[wA,0],[wA,hA],[0,hA]]).reshape(-1,1,2)))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = [-xmin, -ymin]
    T = np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])
    result = cv2.warpPerspective(B, T @ H, (xmax - xmin, ymax - ymin))
    result[translation[1]:translation[1]+hA, translation[0]:translation[0]+wA] = A
    return result


def compute_exposure_gain(A, B, H):
    """
    Tính gain cho B sao cho mean của B(overlap) ~ mean của A(overlap)
    """
    hA, wA = A.shape[:2]
    hB, wB = B.shape[:2]

    maskB = np.ones((hB, wB), dtype=np.uint8) * 255
    maskB_warp = cv2.warpPerspective(maskB, H, (wA, hA))

    overlap = (maskB_warp > 0)
    if overlap.sum() < 100:
        return 1.0

    B_warp = cv2.warpPerspective(B, H, (wA, hA))

    A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    B_gray = cv2.cvtColor(B_warp, cv2.COLOR_BGR2GRAY)

    meanA = A_gray[overlap].mean()
    meanB = B_gray[overlap].mean()

    if meanB < 1e-5:
        return 1.0

    gain = meanA / meanB
    return gain


def apply_gain(B, gain):
    Bf = B.astype(np.float32) * gain
    return np.clip(Bf, 0, 255).astype(np.uint8)


def _image_response(img: np.ndarray):
    _, buf = cv2.imencode('.png', img)
    return send_file(io.BytesIO(buf.tobytes()), mimetype='image/png')


if __name__ == '__main__':
    print('Starting Flask app (no reloader) on 127.0.0.1:5001')
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
