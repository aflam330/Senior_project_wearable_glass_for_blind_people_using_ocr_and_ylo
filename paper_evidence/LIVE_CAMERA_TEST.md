# Live camera loop test

Run 2026-09-28 on the development laptop's built-in webcam (camera 0, DirectShow, 640x480),
models on the RTX 3050 GPU. Script: `savior_glass/scripts/test_live_camera.py`; raw:
`savior_glass/results/live_camera_test.json`. Every live frame went through the glass's own
`CurrencyMode.detect_live` (Taka YOLO + jaal check) and `ObjectMode.process_frame`; every
5th frame through the emotion detector. No frame, image or audio was saved.

| Measure | Result |
|---|---|
| Run length | 20.1 s, 150 frames processed |
| Loop rate | 7.5 frames per second |
| Camera capture | median 73.6 ms (p95 95.6) |
| Taka detection + jaal check | median 31.1 ms (p95 33.0) |
| Object mode | median 23.8 ms (p95 25.6) |
| Emotion (face detection + classifier) | median 18.2 ms (p95 19.6) |
| Whole frame | median 128.9 ms (p95 161.7) |
| Scene brightness | mean 18 / 255 (dark room) |
| Detections | none: no note, object or face was in view |

The camera capture, not the models, limits the loop: in a dark room the webcam lengthens its
exposure. The pipeline ran without errors on real frames.

**NOT_MEASURED:**
- Accuracy on live notes. The run was unattended, so nobody held a note in front of the camera.
- End-to-end latency on the glass's own Pi camera.
- Spoken output during the live run. Speech was tested separately; see `SPEECH_TEST.md`.
- Haptics on real hardware; see `HAPTICS_TEST.md`.
- A demo video, which was not recorded because the webcam faces the user.
