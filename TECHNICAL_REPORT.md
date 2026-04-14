# Technical Report — DetectFlow

## Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage

---

## 1. Model / Detector Used

**YOLOv8m (Medium)** from Ultralytics was selected as the object detector.

YOLOv8 is a single-stage anchor-free detector that achieves state-of-the-art accuracy on the COCO benchmark while maintaining real-time inference speeds. The "medium" variant (YOLOv8m) was chosen as the best balance between accuracy and speed — it achieves significantly higher mAP than the nano/small variants while being substantially faster than the large/xlarge variants.

Key properties:
- **Pre-trained on COCO**: 80 classes including "person" (class 0), which is the primary target for sports footage
- **Anchor-free design**: Better handles objects at varying scales without anchor tuning
- **Multi-scale feature fusion**: Uses a PANet-style neck for detecting both small (distant players) and large (close-up) subjects
- **Input resolution**: 1280px for higher detection accuracy on wide-angle sports footage

The detection confidence threshold is set to 0.3 (intentionally low) because ByteTrack's two-stage association specifically leverages low-confidence detections to recover partially occluded or motion-blurred objects that would be missed at higher thresholds.

## 2. Tracking Algorithm Used

**ByteTrack** is the default multi-object tracking algorithm.

ByteTrack (Zhang et al., ECCV 2022) is a simple yet highly effective tracker that achieves state-of-the-art performance on MOT benchmarks. Its key innovation is the use of **every detection box** — not just high-confidence ones — through a two-stage association process:

1. **First association**: High-confidence detections are matched to existing tracks using IoU-based similarity and the Hungarian algorithm
2. **Second association**: Remaining unmatched tracks are matched against low-confidence detections, recovering objects that might be partially occluded or blurred

The tracker uses a **Kalman filter** for motion prediction, which estimates the next position of each track based on constant-velocity motion. This handles:
- Brief occlusions (predicting position when detection is lost)
- Smooth trajectory estimation
- Noise reduction in detection coordinates

**Track lifecycle management**:
- **New track creation**: Detections unmatched to any existing track create new tracks with new IDs
- **Track buffer**: Lost tracks are kept alive for 30 frames (configurable) before deletion, allowing re-association after brief occlusions
- **Minimum track length**: Tracks shorter than 3 frames are filtered as spurious

**BoT-SORT** is available as an alternative, adding camera-motion compensation (via sparse optical flow) and optional Re-ID feature matching for stronger appearance-based association.

## 3. Why This Combination

The YOLOv8 + ByteTrack combination was selected for several reasons:

1. **Integrated implementation**: Ultralytics provides ByteTrack and BoT-SORT directly integrated with YOLOv8's detection pipeline, ensuring optimal information flow between detection and tracking stages

2. **ByteTrack's low-confidence recovery**: In sports footage, players frequently overlap, are partially occluded by other players, or appear blurred during rapid movement. ByteTrack's two-stage association explicitly handles these cases by utilizing all detections

3. **No Re-ID model required**: ByteTrack achieves competitive performance using only motion and IoU cues, without requiring a separate appearance model. This simplifies deployment and reduces computational cost

4. **Proven track record**: This combination consistently ranks among the top approaches on MOT17 and MOT20 benchmarks, which feature scenarios similar to sports footage (crowds, occlusions, varying scales)

5. **Practical considerations**: The pipeline runs on CPU (though GPU is recommended), uses pre-trained weights (no custom training needed), and handles standard video formats

## 4. How ID Consistency Is Maintained

ID consistency is maintained through multiple mechanisms:

1. **Kalman filter prediction**: When a detection is temporarily lost, the Kalman filter predicts where the object should be. When the detection reappears, it's associated with the predicted position, preserving the ID

2. **IoU-based matching**: The Hungarian algorithm finds the optimal assignment between predicted track positions and new detections based on IoU overlap, ensuring the closest spatial match gets the same ID

3. **Two-stage association**: Low-confidence detections (which often occur during occlusion or blur) are matched in the second stage to tracks that weren't matched in the first stage, reducing unnecessary ID switches

4. **Track buffer**: Tracks aren't immediately deleted when lost — they persist for up to 30 frames, allowing re-association if the object reappears

5. **Motion model**: The constant-velocity Kalman filter handles linear motion well, and the track buffer handles non-linear motion during brief occlusions

## 5. Challenges Faced

- **Similar appearance**: In team sports, players wearing the same uniform are difficult to distinguish by appearance alone. ByteTrack mitigates this through strong motion cues, but ID switches can still occur during close interactions

- **Camera motion**: Rapid panning or zooming affects all detections simultaneously. BoT-SORT's camera-motion compensation (available as an option) addresses this via sparse optical flow

- **Scale variation**: Players near the camera appear much larger than those far away. YOLOv8's multi-scale feature pyramid handles this, but very distant players may still be missed

- **Crowded scenes**: Dense groups of players (e.g., during set pieces) cause significant occlusion, leading to missed detections and potential ID switches

- **Processing speed**: CPU inference at 1280px resolution is slow (~2-5 FPS). Frame skipping can be used to trade temporal resolution for speed

## 6. Failure Cases Observed

- **Extended occlusion** (>30 frames): When a player is fully occluded for longer than the track buffer, they receive a new ID upon reappearing

- **Close interactions**: Two players crossing paths or colliding at similar speeds can cause ID swaps, especially if their bounding boxes overlap significantly

- **Abrupt camera cuts**: Scene transitions completely break tracking continuity, as the Kalman filter's predictions become invalid

- **Very small subjects**: Players at extreme distances (a few pixels tall) may not be detected consistently

- **Re-entry from frame boundary**: A player leaving and re-entering the frame gets a new ID, as the tracker has no mechanism for out-of-frame persistence

## 7. Possible Improvements

1. **Re-ID features**: Adding a lightweight appearance embedding model (e.g., OSNet) would allow the tracker to re-identify players after longer occlusions based on visual similarity

2. **Camera-motion compensation**: Using BoT-SORT with GMC (Global Motion Compensation) via optical flow would improve tracking during rapid camera movement

3. **Sport-specific fine-tuning**: Fine-tuning YOLOv8 on sport-specific datasets (e.g., SoccerNet) would improve detection accuracy for players in specific sports

4. **Team classification**: Adding a secondary classifier to distinguish teams based on jersey color would enable team-specific analytics

5. **Homography estimation**: Mapping detections to a top-down field/court view would enable tactical analysis, speed estimation, and formation detection

6. **Temporal smoothing**: Post-processing track trajectories with spline interpolation would fill gaps during brief detection losses

7. **Online learning**: Adapting appearance features during tracking to handle changing lighting conditions and player poses

---

*Report prepared as part of the DetectFlow multi-object tracking pipeline.*
