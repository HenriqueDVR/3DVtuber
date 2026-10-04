"""Where an anime face detector puts its 28 landmarks on a picture, as JSON. Plain Python with PyTorch.

    python scripts/retexture/guess_face_points.py picture.png

Needs `pip install anime-face-detector` (hysts, MIT; pure PyTorch, Python 3.12+; 273 MB of
weights from the Hugging Face Hub on first use). On the CPU: about 10 s to load, 1 s a picture.
The app turns the landmarks into face points (app/face_guess.py); this only runs the detector,
in whichever Python has it (the app's own needn't).
Prints `FACE_LANDMARKS {"size": [w, h], "faces": [{"box": [x1, y1, x2, y2, score], "points": [[x, y, score] * 28]}]}`.
"""
import json
import sys

import cv2
from anime_face_detector import create_detector

image = cv2.imread(sys.argv[1])
if image is None:
    sys.exit(f"can't read {sys.argv[1]}")
faces = create_detector("yolov3", device="cpu")(image)
print("FACE_LANDMARKS " + json.dumps({
    "size": [image.shape[1], image.shape[0]],
    "faces": [{"box": [round(float(v), 2) for v in face["bbox"]],
               "points": [[round(float(v), 2) for v in point] for point in face["keypoints"]]} for face in faces]}))
