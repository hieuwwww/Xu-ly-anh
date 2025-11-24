# Panorama Step Viewer

Local Flask app to upload images and inspect panorama stitching steps.

How to run (Windows PowerShell):

```powershell
python -m pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000 in browser
```

Usage:
- Upload 2+ images
- View keypoints, matches, warp B->A, and run simple custom stitch
