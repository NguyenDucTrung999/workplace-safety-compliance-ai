import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ultralytics import YOLO
from config import HUMAN_MODEL_PATH, DEVICE

class HumanDetectorOOP:
    def __init__(self, weights_path=None, confidence_threshold=0.50, image_size=320, device=None):
        if weights_path is None:
            weights_path = HUMAN_MODEL_PATH
        else:
            weights_path = str(weights_path)
            
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        if device is not None:
            self.device = device
        else:
            self.device = DEVICE
        
        engine_weights_path = self.weights_path.replace(".pt", ".engine")
        if os.path.exists(engine_weights_path):
            print(f"Phát hiện TensorRT Engine Model 1: {engine_weights_path}, tiến hành nạp Engine...")
            self.weights_path = engine_weights_path

        print(f"Đang tải Model 1 Human Detection từ: {self.weights_path} trên thiết bị: {self.device}")
        self.model = YOLO(self.weights_path).to(self.device)

        self.is_cuda_device = (
            (hasattr(self.device, 'type') and self.device.type == 'cuda') or
            (isinstance(self.device, str) and 'cuda' in self.device)
        )

    def predict(self, frame):
        if frame is None or frame.size == 0:
            return []
            
        prediction_results = self.model.predict(
            source=frame, 
            conf=self.confidence_threshold, 
            imgsz=self.image_size, 
            device=self.device,
            half=self.is_cuda_device,
            verbose=False
        )
        
        first_result = prediction_results[0]
        detected_boxes_list = []
        
        for box in first_result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            box_confidence = float(box.conf[0])
            
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = None
                
            detected_boxes_list.append([x_min, y_min, x_max, y_max, box_confidence, track_id])
            
        return detected_boxes_list

    def reset(self):
        if hasattr(self.model, 'predictor') and self.model.predictor is not None:
            if hasattr(self.model.predictor, 'trackers'):
                for tracker in self.model.predictor.trackers:
                    tracker.reset()
