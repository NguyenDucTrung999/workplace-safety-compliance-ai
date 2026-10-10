import math
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import DEVICE

class DeepFeatureExtractorOOP:
    def __init__(self, device=None):
        if device is None:
            self.device = DEVICE
        else:
            self.device = torch.device(device)

        print(f"Đang khởi tạo Deep Re-ID Feature Extractor (MobileNetV3) trên thiết bị: {self.device}")
        
        # Tải mô hình MobileNetV3 Small trích xuất vector đặc trưng 512 chiều
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        base_model.classifier = torch.nn.Identity()
        base_model.to(self.device)

        self.is_cuda_device = (
            (hasattr(self.device, 'type') and self.device.type == 'cuda') or
            (isinstance(self.device, str) and 'cuda' in self.device)
        )

        dtype = torch.float16 if self.is_cuda_device else torch.float32
        if self.is_cuda_device:
            base_model.half()

        self.mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=dtype).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=dtype).view(1, 3, 1, 1)

        base_model.eval()
        try:
            sample_dtype = torch.float16 if self.is_cuda_device else torch.float32
            dummy_input = torch.zeros((1, 3, 128, 64), device=self.device, dtype=sample_dtype)
            traced_model = torch.jit.trace(base_model, dummy_input)
            traced_model = torch.jit.optimize_for_inference(traced_model)
            self.feature_extractor = traced_model
            print("Đã tối ưu hóa Deep Re-ID bằng TorchScript JIT thành công.")
        except Exception as jit_err:
            print(f"Cảnh báo: Không thể JIT trace Re-ID, chạy fallback native: {jit_err}")
            self.feature_extractor = base_model

    def extract_feature_vector(self, crop_image):
        if crop_image is None or crop_image.size == 0 or crop_image.shape[0] < 5 or crop_image.shape[1] < 5:
            return None

        resized_crop = cv2.resize(crop_image, (64, 128), interpolation=cv2.INTER_LINEAR)
        rgb_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
        tensor_input = torch.from_numpy(rgb_crop).permute(2, 0, 1).unsqueeze(0)

        if self.is_cuda_device:
            tensor_input = tensor_input.to(self.device, non_blocking=True).half() / 255.0
        else:
            tensor_input = tensor_input.to(self.device, non_blocking=True).float() / 255.0

        tensor_input = (tensor_input - self.mean_tensor) / self.std_tensor

        with torch.no_grad():
            feature_embedding = self.feature_extractor(tensor_input)
            l2_normalized_vector = torch.nn.functional.normalize(feature_embedding, p=2, dim=1)
            numpy_vector = l2_normalized_vector.cpu().numpy()[0]
            
        return numpy_vector

    def calculate_cosine_similarity(self, vector_a, vector_b):
        if vector_a is None or vector_b is None:
            return 0.0
        similarity_score = float(np.dot(vector_a, vector_b))
        return similarity_score


class TrackerEngineOOP:
    def __init__(self, max_disappeared_frames=3600, iou_threshold=0.25, distance_threshold=350.0, reid_similarity_threshold=0.45):
        self.next_track_id = 1
        self.frame_counter = 0
        self.tracked_objects_dictionary = {}
        self.disappeared_frames_dictionary = {}
        self.person_feature_gallery = {}
        self.max_disappeared_frames = max_disappeared_frames
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.reid_similarity_threshold = reid_similarity_threshold

        try:
            self.deep_reid_extractor = DeepFeatureExtractorOOP()
        except Exception as error_msg:
            print(f"Cảnh báo: Không thể khởi tạo Deep Re-ID Extractor: {str(error_msg)}")
            self.deep_reid_extractor = None

    def calculate_intersection_over_union(self, box_a, box_b):
        x_a_min, y_a_min, x_a_max, y_a_max = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
        x_b_min, y_b_min, x_b_max, y_b_max = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

        intersection_x_min = max(x_a_min, x_b_min)
        intersection_y_min = max(y_a_min, y_b_min)
        intersection_x_max = min(x_a_max, x_b_max)
        intersection_y_max = min(y_a_max, y_b_max)

        intersection_width = max(0.0, intersection_x_max - intersection_x_min)
        intersection_height = max(0.0, intersection_y_max - intersection_y_min)
        intersection_area = intersection_width * intersection_height

        area_a = max(0.0, (x_a_max - x_a_min) * (y_a_max - y_a_min))
        area_b = max(0.0, (x_b_max - x_b_min) * (y_b_max - y_b_min))
        union_area = area_a + area_b - intersection_area

        if union_area > 0.0:
            return float(intersection_area / union_area)
        else:
            return 0.0

    def calculate_centroid(self, box):
        x_min, y_min, x_max, y_max = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        x_center = float((x_min + x_max) / 2.0)
        y_center = float((y_min + y_max) / 2.0)
        return (x_center, y_center)

    def update(self, detected_boxes, frame=None):
        self.frame_counter += 1
        if len(detected_boxes) == 0:
            for track_id in list(self.disappeared_frames_dictionary.keys()):
                self.disappeared_frames_dictionary[track_id] += 1
                if self.disappeared_frames_dictionary[track_id] > self.max_disappeared_frames:
                    if track_id in self.tracked_objects_dictionary:
                        del self.tracked_objects_dictionary[track_id]
                    del self.disappeared_frames_dictionary[track_id]
            return []

        results_list = []
        used_track_ids_set = set()

        for box in detected_boxes:
            if len(box) >= 6:
                x_min, y_min, x_max, y_max, confidence, byte_track_id = box[:6]
            else:
                x_min, y_min, x_max, y_max, confidence = box[:5]
                byte_track_id = None

            current_box = [x_min, y_min, x_max, y_max]
            current_centroid = self.calculate_centroid(current_box)
            assigned_track_id = None

            if assigned_track_id is None and byte_track_id is not None:
                b_id = int(byte_track_id)
                if b_id not in used_track_ids_set:
                    assigned_track_id = b_id

            if assigned_track_id is None:
                best_iou = 0.0
                best_iou_track_id = None

                for existing_track_id, existing_data in self.tracked_objects_dictionary.items():
                    if existing_track_id in used_track_ids_set:
                        continue
                    
                    existing_box = existing_data["box"]
                    iou_score = self.calculate_intersection_over_union(current_box, existing_box)
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_iou_track_id = existing_track_id

                if best_iou >= self.iou_threshold:
                    assigned_track_id = best_iou_track_id

            feature_vector = None
            should_extract_reid = False
            if assigned_track_id is None:
                should_extract_reid = True
            elif assigned_track_id not in self.person_feature_gallery:
                should_extract_reid = True
            else:
                track_id_numeric = 0
                try:
                    track_id_numeric = int(assigned_track_id)
                except (ValueError, TypeError):
                    track_id_numeric = 0

                should_extract_reid = ((self.frame_counter + track_id_numeric) % 25 == 0)

            if should_extract_reid and frame is not None and self.deep_reid_extractor is not None:
                img_h, img_w = frame.shape[:2]
                c_xmin, c_ymin = max(0, int(x_min)), max(0, int(y_min))
                c_xmax, c_ymax = min(img_w, int(x_max)), min(img_h, int(y_max))
                if c_xmax > c_xmin and c_ymax > c_ymin:
                    crop = frame[c_ymin:c_ymax, c_xmin:c_xmax]
                    feature_vector = self.deep_reid_extractor.extract_feature_vector(crop)

            if assigned_track_id is None and feature_vector is not None and len(self.person_feature_gallery) > 0:
                best_similarity = 0.0
                best_similarity_track_id = None

                for gallery_track_id, gallery_vector in self.person_feature_gallery.items():
                    if gallery_track_id in used_track_ids_set:
                        continue
                    sim_score = self.deep_reid_extractor.calculate_cosine_similarity(feature_vector, gallery_vector)
                    if sim_score > best_similarity:
                        best_similarity = sim_score
                        best_similarity_track_id = gallery_track_id

                if best_similarity >= self.reid_similarity_threshold:
                    assigned_track_id = best_similarity_track_id

            if assigned_track_id is None:
                best_distance = float("inf")
                best_distance_track_id = None

                for existing_track_id, existing_data in self.tracked_objects_dictionary.items():
                    if existing_track_id in used_track_ids_set:
                        continue

                    existing_centroid = existing_data["centroid"]
                    distance = math.hypot(current_centroid[0] - existing_centroid[0], current_centroid[1] - existing_centroid[1])

                    if distance < best_distance:
                        best_distance = distance
                        best_distance_track_id = existing_track_id

                if best_distance < self.distance_threshold and best_distance_track_id is not None:
                    assigned_track_id = best_distance_track_id

            if assigned_track_id is None:
                assigned_track_id = self.next_track_id
                self.next_track_id += 1

            self.tracked_objects_dictionary[assigned_track_id] = {
                "box": current_box,
                "centroid": current_centroid,
                "confidence": float(confidence)
            }
            self.disappeared_frames_dictionary[assigned_track_id] = 0
            used_track_ids_set.add(assigned_track_id)

            if feature_vector is not None:
                if assigned_track_id in self.person_feature_gallery:
                    old_vector = self.person_feature_gallery[assigned_track_id]
                    updated_vector = 0.8 * old_vector + 0.2 * feature_vector
                    normalized_updated_vector = updated_vector / (np.linalg.norm(updated_vector) + 1e-6)
                    self.person_feature_gallery[assigned_track_id] = normalized_updated_vector
                else:
                    self.person_feature_gallery[assigned_track_id] = feature_vector

            results_list.append([assigned_track_id, int(x_min), int(y_min), int(x_max), int(y_max), float(confidence)])

        all_active_track_ids = list(self.tracked_objects_dictionary.keys())
        for existing_track_id in all_active_track_ids:
            if existing_track_id not in used_track_ids_set:
                self.disappeared_frames_dictionary[existing_track_id] = self.disappeared_frames_dictionary.get(existing_track_id, 0) + 1
                if self.disappeared_frames_dictionary[existing_track_id] > self.max_disappeared_frames:
                    del self.tracked_objects_dictionary[existing_track_id]
                    del self.disappeared_frames_dictionary[existing_track_id]

        return results_list

    def reset(self):
        self.next_track_id = 1
        self.tracked_objects_dictionary.clear()
        self.disappeared_frames_dictionary.clear()
        self.person_feature_gallery.clear()
