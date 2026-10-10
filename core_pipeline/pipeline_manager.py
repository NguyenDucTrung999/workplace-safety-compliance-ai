import cv2
import sys
import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Đảm bảo đường dẫn import hoạt động chuẩn
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core_pipeline.human_detector import HumanDetectorOOP
from tracking.tracker_engine import TrackerEngineOOP
from core_pipeline.ppe_classifier import PPEClassifierOOP
from core_pipeline.state_manager import SafetyStateManagerOOP
from core_pipeline.roi_manager import ROIManagerOOP

VIOLATION_VN_MAP = {
    "no_helmet": "Thiếu mũ",
    "no_vest": "Thiếu áo",
    "no_boots": "Thiếu giày",
    "no_id_card": "Thiếu thẻ",
    "smoking": "Hút thuốc",
    "phone_usage": "Dùng điện thoại",
    "UNAUTHORIZED_INTRUSION": "Sai vùng"
}

_FONT_CACHE = {}

def get_vietnamese_font(font_pixel_size=15):
    if font_pixel_size in _FONT_CACHE:
        return _FONT_CACHE[font_pixel_size]

    candidate_font_paths = [
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    selected_font = None
    for p in candidate_font_paths:
        if os.path.exists(p):
            try:
                selected_font = ImageFont.truetype(p, font_pixel_size)
                break
            except Exception:
                continue

    if selected_font is None:
        try:
            selected_font = ImageFont.load_default()
        except Exception:
            selected_font = None

    _FONT_CACHE[font_pixel_size] = selected_font
    return selected_font

def draw_vietnamese_text_pil(frame, text_content, text_position, text_color_bgr=(0, 255, 0), font_pixel_size=15):
    font_style = get_vietnamese_font(font_pixel_size)
    text_x, text_y = text_position
    frame_h, frame_w = frame.shape[:2]

    # Tính toán kích thước chữ trực tiếp từ font_style, loại bỏ đối tượng trung gian
    if hasattr(font_style, 'getbbox'):
        b_x1, b_y1, b_x2, b_y2 = font_style.getbbox(text_content)
    else:
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        b_x1, b_y1, b_x2, b_y2 = dummy_draw.textbbox((0, 0), text_content, font=font_style)

    text_width = b_x2 - b_x1 + 10
    text_height = b_y2 - b_y1 + 8

    crop_x1 = max(0, int(text_x - 2))
    crop_y1 = max(0, int(text_y - 2))
    crop_x2 = min(frame_w, crop_x1 + text_width)
    crop_y2 = min(frame_h, crop_y1 + text_height)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return frame

    sub_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    rgb_sub_crop = cv2.cvtColor(sub_crop, cv2.COLOR_BGR2RGB)
    pil_sub_image = Image.fromarray(rgb_sub_crop)
    draw_canvas = ImageDraw.Draw(pil_sub_image)

    bg_box = (0, 0, crop_x2 - crop_x1, crop_y2 - crop_y1)
    draw_canvas.rectangle(bg_box, fill=(18, 24, 36))
    text_color_rgb = (text_color_bgr[2], text_color_bgr[1], text_color_bgr[0])
    draw_canvas.text((2, 2), text_content, font=font_style, fill=text_color_rgb)

    bgr_sub_result = cv2.cvtColor(np.array(pil_sub_image), cv2.COLOR_RGB2BGR)
    frame[crop_y1:crop_y2, crop_x1:crop_x2] = bgr_sub_result
    return frame

class SafetyPipelineManagerOOP:
    def __init__(self, human_weights_path=None, ppe_weights_path=None, padding_ratio=0.10):
        print("=== BẮT ĐẦU KHỞI TẠO SAFETY PIPELINE MANAGER OOP ===")
        self.human_detector = HumanDetectorOOP(weights_path=human_weights_path)
        self.tracker_engine = TrackerEngineOOP()
        self.ppe_classifier = PPEClassifierOOP(weights_path=ppe_weights_path)
        self.state_manager = SafetyStateManagerOOP()
        self.roi_manager = ROIManagerOOP()
        self.padding_ratio = padding_ratio
        
        # Tối ưu hóa hiệu năng Jetson Orin Nano: Cache kết quả Model 2 PPE Classifier
        self.ppe_detections_cache_dictionary = {}
        self.frame_counter = 0
        print("=== THIẾT LẬP PIPELINE THÀNH CÔNG ===")

    def crop_person_with_padding(self, frame, x_min, y_min, x_max, y_max):
        image_height, image_width = frame.shape[:2]
        box_width = x_max - x_min
        box_height = y_max - y_min

        padding_x = box_width * self.padding_ratio
        padding_y = box_height * self.padding_ratio

        padded_x_min = max(0, int(x_min - padding_x))
        padded_y_min = max(0, int(y_min - padding_y))
        padded_x_max = min(image_width, int(x_max + padding_x))
        padded_y_max = min(image_height, int(y_max + padding_y))

        cropped_image = frame[padded_y_min:padded_y_max, padded_x_min:padded_x_max]
        return cropped_image, (padded_x_min, padded_y_min, padded_x_max, padded_y_max)

    def process_frame(self, frame, current_timestamp=None):
        if frame is None or frame.size == 0:
            return None, {"error": "Khung hình rỗng"}, []

        if current_timestamp is None:
            current_timestamp = time.time()
        else:
            current_timestamp = float(current_timestamp)

        self.frame_counter += 1

        annotated_frame = frame.copy()
        
        detected_human_boxes = self.human_detector.predict(frame)
        tracked_persons = self.tracker_engine.update(detected_human_boxes, frame)
        
        persons_detail_list = []
        image_height, image_width = frame.shape[:2]

        cropped_images_batch = []
        batch_index_map = {}
        person_preprocessed_list = []
        active_track_ids_this_frame = set()

        for tracked_person in tracked_persons:
            track_id, x_min, y_min, x_max, y_max, confidence = tracked_person
            active_track_ids_this_frame.add(track_id)
            bounding_box = [int(x_min), int(y_min), int(x_max), int(y_max)]

            edge_margin_pixels = 15
            is_boundary_cutoff = (
                int(x_min) <= edge_margin_pixels or 
                int(y_min) <= edge_margin_pixels or 
                int(x_max) >= (image_width - edge_margin_pixels) or 
                int(y_max) >= (image_height - edge_margin_pixels)
            )
            
            cropped_image, _ = self.crop_person_with_padding(frame, x_min, y_min, x_max, y_max)
            cropped_images_batch.append(cropped_image)
            batch_index_map[len(cropped_images_batch) - 1] = track_id

            person_preprocessed_list.append({
                "track_id": track_id,
                "bounding_box": bounding_box,
                "confidence": float(confidence),
                "is_boundary_cutoff": is_boundary_cutoff
            })

        if len(cropped_images_batch) > 0:
            batch_ppe_detections = self.ppe_classifier.predict_batch(cropped_images_batch)
            for batch_idx, ppe_dets in enumerate(batch_ppe_detections):
                target_track_id = batch_index_map[batch_idx]
                self.ppe_detections_cache_dictionary[target_track_id] = ppe_dets

        for person_preprocessed in person_preprocessed_list:
            track_id = person_preprocessed["track_id"]
            bounding_box = person_preprocessed["bounding_box"]
            confidence = person_preprocessed["confidence"]
            is_boundary_cutoff = person_preprocessed["is_boundary_cutoff"]
            
            ppe_detections = self.ppe_detections_cache_dictionary.get(track_id, [])

            roi_evaluations = self.roi_manager.evaluate_person_roi(
                track_id=track_id, 
                bounding_box=bounding_box, 
                ppe_detections_list=ppe_detections
            )
            
            person_info = {
                "track_id": track_id,
                "bounding_box": bounding_box,
                "confidence": confidence,
                "ppe_detections": ppe_detections,
                "roi_evaluations": roi_evaluations,
                "is_boundary_cutoff": is_boundary_cutoff
            }
            persons_detail_list.append(person_info)

        state_summary, new_events_list = self.state_manager.update(
            persons_detail_list=persons_detail_list,
            current_timestamp=current_timestamp,
            original_frame=frame
        )

        for person_info in persons_detail_list:
            track_id = person_info["track_id"]
            x_min, y_min, x_max, y_max = person_info["bounding_box"]
            roi_evaluations = person_info["roi_evaluations"]
            is_boundary_cutoff = person_info.get("is_boundary_cutoff", False)

            display_name = self.roi_manager.get_person_display_name(track_id)
            stacked_violations_list = []

            if track_id in self.state_manager.tracked_person_states:
                person_state = self.state_manager.tracked_person_states[track_id]
                for v_name in person_state.active_violations_dictionary.keys():
                    stacked_violations_list.append(v_name)

            for roi_eval in roi_evaluations:
                for z_viol in roi_eval["zone_violations"]:
                    stacked_violations_list.append(z_viol["violation_type"])

            prefix_id_text = f"ID: {track_id}"
            if display_name and display_name != f"ID: {track_id}":
                prefix_id_text = display_name

            if len(stacked_violations_list) > 0:
                box_color = (0, 255, 255)
                missing_texts = [VIOLATION_VN_MAP.get(viol, viol) for viol in stacked_violations_list]
                violations_string = ", ".join(missing_texts)
                label_text = f"{prefix_id_text} | {violations_string}"
            elif is_boundary_cutoff:
                box_color = (180, 180, 180)
                label_text = f"{prefix_id_text} | Cắt mép cam"
            else:
                box_color = (0, 255, 0)
                label_text = f"{prefix_id_text}"

            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), box_color, 2)
            annotated_frame = draw_vietnamese_text_pil(
                frame=annotated_frame,
                text_content=label_text,
                text_position=(x_min, max(0, y_min - 22)),
                text_color_bgr=box_color,
                font_pixel_size=14
            )

        current_frame_track_ids = [int(tracked_item[0]) for tracked_item in tracked_persons]

        pipeline_report_dictionary = {
            "total_tracked_persons": len(tracked_persons),
            "current_frame_track_ids": current_frame_track_ids,
            "persons_detail": persons_detail_list,
            "state_summary": state_summary
        }

        return annotated_frame, pipeline_report_dictionary, new_events_list

    def reset(self):
        self.human_detector.reset()
        self.tracker_engine.reset()
        self.state_manager.reset()
