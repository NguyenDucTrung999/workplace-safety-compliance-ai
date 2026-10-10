import cv2
import numpy as np
from core_pipeline.human_detector import HumanDetectorOOP

detector = HumanDetectorOOP()
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Dummy test frame to run inference
# Use random noise instead of solid white to trigger something maybe?
# Let's just check if predictor has trackers
res1 = detector.predict(frame)
if hasattr(detector.model, 'predictor') and detector.model.predictor is not None:
    if hasattr(detector.model.predictor, 'trackers'):
        for tracker in detector.model.predictor.trackers:
            tracker.reset()
            print("Tracker reset successfully!")
