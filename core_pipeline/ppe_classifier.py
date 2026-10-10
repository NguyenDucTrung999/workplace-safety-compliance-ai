import os
import sys
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ultralytics import YOLO
from config import PPE_MODEL_PATH, DEVICE

PPE_CONFIDENCE_THRESHOLDS = {
    "phone_usage": 0.60,
    "smoking": 0.60,
    "boots": 0.20,
    "no boots": 0.20,
    "id_card": 0.25,
    "IDcard": 0.25,
    "ID_card": 0.25,
}
DEFAULT_PPE_CONFIDENCE = 0.40

class PPEClassifierOOP:
    def __init__(self, weights_path=None, confidence_threshold=0.18, image_size=640, device=None):
        if weights_path is None:
            weights_path = PPE_MODEL_PATH
        else:
            weights_path = str(weights_path)
            
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device if device is not None else DEVICE
        
        engine_weights_path = self.weights_path.replace(".pt", ".engine")
        if os.path.exists(engine_weights_path):
            print(f"Phát hiện TensorRT Engine Model 2 PPE: {engine_weights_path}, tiến hành nạp Engine...")
            self.weights_path = engine_weights_path

        print(f"Đang tải Model 2 PPE Detection từ: {self.weights_path} trên thiết bị: {self.device}")
        self.model = YOLO(self.weights_path).to(self.device)

        self.is_cuda_device = (
            (hasattr(self.device, 'type') and self.device.type == 'cuda') or
            (isinstance(self.device, str) and 'cuda' in self.device)
        )

    def predict(self, cropped_person_image):
        if cropped_person_image is None or cropped_person_image.size == 0:
            return []
        batch_results = self.predict_batch([cropped_person_image])
        return batch_results[0] if batch_results else []

    def predict_batch(self, cropped_images_list):
        if not cropped_images_list:
            return []

        valid_images_list = [
            img if (img is not None and img.size > 0) else np.zeros((64, 64, 3), dtype=np.uint8)
            for img in cropped_images_list
        ]

        prediction_results = self.model.predict(
            source=valid_images_list,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            half=self.is_cuda_device,
            verbose=False
        )

        all_crops_detected_ppe_list = []

        for index, single_result in enumerate(prediction_results):
            crop_height, crop_width = valid_images_list[index].shape[:2]
            single_crop_ppe_list = []

            for box in single_result.boxes:
                class_id = int(box.cls[0])
                class_name = single_result.names[class_id]
                box_confidence = float(box.conf[0])
                box_x_min, box_y_min, box_x_max, box_y_max = box.xyxy[0].tolist()

                minimum_required_confidence = PPE_CONFIDENCE_THRESHOLDS.get(class_name, DEFAULT_PPE_CONFIDENCE)

                if box_confidence >= minimum_required_confidence:
                    relative_y_center = float((box_y_min + box_y_max) / (2.0 * crop_height)) if crop_height > 0 else 0.5

                    single_crop_ppe_list.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": box_confidence,
                        "box": [float(box_x_min), float(box_y_min), float(box_x_max), float(box_y_max)],
                        "relative_y_center": relative_y_center
                    })

            all_crops_detected_ppe_list.append(single_crop_ppe_list)

        return all_crops_detected_ppe_list
