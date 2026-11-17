# Person-tracking drone (DJI Tello)

A small DJI Tello drone that finds one specific person in its camera feed and
keeps them centred by moving left, right, up, down, forward or back. The intended
use is search and rescue: given a photo of a missing person, the drone locks on
when it sees them.

This is a student prototype built in my second year, tested indoors.

## How it works

```
Tello camera ──► frame (OpenCV)
                   │
                   ├─► face_recognition: find faces, compare each with known_face.jpg
                   │        └─ match? ─► steer so the face sits in the centre of the frame
                   │                     and keeps a target size (distance)
                   │
                   └─► small CNN (64×64 input): "is the target in view?"
                            └─ no face found but CNN says yes ─► move forward to search
```

- **Recognition.** [`face_recognition`](https://github.com/ageitgey/face_recognition)
  encodes the reference photo `known_face.jpg` and every face in each frame, and
  compares them.
- **Steering.** The offset between the face centre and the frame centre drives
  20 cm moves once it exceeds 30 pixels. Face height is used as a distance cue:
  the drone moves closer or backs off to keep it near 100 pixels.
- **Fallback detector.** A two-layer CNN trained on your own images
  (`train_model.py`) gives a coarse "target in view" signal when no face is
  visible, for example when the person is facing away.

## Files

| File | What it does |
|---|---|
| `drone_inference.py` | Main loop: connects to the Tello, recognises the target, steers the drone |
| `train_model.py` | Trains the 64×64 binary CNN and saves `target_model.h5` |
| `video_capture.py` | Shows a webcam feed, useful for checking the camera before collecting data |
| `intergrated_file_test.py` | Earlier end-to-end version: captures frames, trains the CNN, then flies on the CNN alone |

## Run it

1. Install the dependencies. `face_recognition` needs dlib, which needs CMake and a C++ compiler.

   ```bash
   pip install -r requirements.txt
   ```

2. Put training images in `data/train/<class>/` and held-out images in
   `data/validation/<class>/`, with one folder for the target and one for
   everything else. Then train:

   ```bash
   python train_model.py
   ```

3. Save a clear, front-facing photo of the person to track as `known_face.jpg`.
4. Connect your computer to the Tello's Wi-Fi and run:

   ```bash
   python drone_inference.py
   ```

   Press `q` in the video window to stop the stream.

## Limitations

- The script does not call `takeoff()` or `land()`. Take off and land the drone
  yourself, and keep a hand on the controller.
- Steering is a fixed-step threshold rule, not a smooth controller, so the drone
  moves in jerky 20 cm steps.
- The CNN is very small and trained on a few images. It is a coarse cue, not a
  reliable detector.
- No accuracy or tracking measurements have been published yet.

## Responsible use

Face recognition identifies real people. Only use photos and footage of people
who have agreed to it, check your local drone and privacy laws before flying, and
do not commit face images or recordings to this repository.
