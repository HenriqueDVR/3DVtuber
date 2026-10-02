# Third-party files kept with the app

What the app's pages would otherwise load from the internet, kept here unmodified so the app
runs with no connection. `app/vendor.py` pins each file by SHA-256 and fetches any that is
missing (`python -m app vendor`); `tests/test_app.py` checks the copies here against the pins.
None of these is part of VTuber Maker or covered by its own licence.

| Files | What | Version | From | Licence |
|---|---|---|---|---|
| `npm/three@0.180.0/` | three.js: the 3D views | 0.180.0 | https://cdn.jsdelivr.net/npm/three@0.180.0/ | MIT |
| `npm/@pixiv/three-vrm@3.5.5/` | three-vrm: loads and runs VRM avatars in the viewer | 3.5.5 | https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@3.5.5/ | MIT |
| `npm/@mediapipe/tasks-vision@0.10.14/` | MediaPipe Tasks Vision: the face tracker of the viewer's webcam test | 0.10.14 | https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/ | Apache-2.0 |
| `mediapipe-models/face_landmarker/` | MediaPipe Face Landmarker model (float16, version 1) | 2023-05-03 | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task | Apache-2.0 |

## three.js

Copyright 2010-2025 Three.js Authors. https://github.com/mrdoob/three.js

## three-vrm

Copyright (c) 2019-2026 pixiv Inc. https://github.com/pixiv/three-vrm

## The MIT License (three.js and three-vrm)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## MediaPipe Tasks Vision and the Face Landmarker model

Copyright The MediaPipe Authors (Google LLC). https://github.com/google-ai-edge/mediapipe ;
model cards: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker

Licensed under the Apache License, Version 2.0: the full text is in `Apache-2.0.txt` beside
this file. Both are kept here exactly as published.
