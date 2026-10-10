import sys
import os
import cv2
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core_pipeline.roi_manager import ROIManagerOOP

def test_roi_manager():
    print("=== BẮT ĐẦU KIỂM THỬ ROI MANAGER OOP ===")
    roi_manager = ROIManagerOOP()

    # Thêm 1 Vùng cấm (RESTRICTED_ZONE)
    restricted_polygon = [(100, 100), (400, 100), (400, 400), (100, 400)]
    roi_manager.add_zone(
        zone_id="ZONE_01",
        zone_name="Khu vuc May Nang Cap A",
        zone_type="RESTRICTED_ZONE",
        polygon_points_list=restricted_polygon
    )

    # Thêm 1 Vùng bắt buộc trang bị bảo hộ (MANDATORY_PPE_ZONE)
    mandatory_ppe_polygon = [(500, 100), (900, 100), (900, 400), (500, 400)]
    roi_manager.add_zone(
        zone_id="ZONE_02",
        zone_name="Khu vuc Thi Cong B",
        zone_type="MANDATORY_PPE_ZONE",
        polygon_points_list=mandatory_ppe_polygon,
        required_ppe_list=["helmet on", "vest"]
    )

    # Đánh giá 1 người đứng trong Vùng Cấm (ZONE_01)
    person1_box = [150, 150, 250, 380] # chân tại (200, 380) -> Trong ZONE_01
    person1_ppe = [{"class_name": "helmet on"}]
    roi_eval1 = roi_manager.evaluate_person_roi(track_id=1, bounding_box=person1_box, ppe_detections_list=person1_ppe)

    print("--------------------------------------------------")
    print(f"Đánh giá Người 1 (Chân tại 200, 380): {roi_eval1}")
    if len(roi_eval1) > 0 and roi_eval1[0]["zone_violations"][0]["violation_type"] == "RESTRICTED_ZONE_INTRUSION":
        print("KIỂM THỬ CHÍNH XÁC: Phát hiện xâm nhập Vùng cấm ZONE_01 thành công!")
    else:
        print(f"KIỂM THỬ CÓ LỖI: {roi_eval1}")

    # Đánh giá 1 người đứng trong Vùng Bắt Buộc Bảo Hộ (ZONE_02) nhưng thiếu Áo
    person2_box = [600, 150, 700, 380] # chân tại (650, 380) -> Trong ZONE_02
    person2_ppe = [{"class_name": "helmet on"}, {"class_name": "no vest"}]
    roi_eval2 = roi_manager.evaluate_person_roi(track_id=2, bounding_box=person2_box, ppe_detections_list=person2_ppe)

    print("--------------------------------------------------")
    print(f"Đánh giá Người 2 (Chân tại 650, 380): {roi_eval2}")
    if len(roi_eval2) > 0 and "MISSING_VEST_IN_ZONE" in [v["violation_type"] for v in roi_eval2[0]["zone_violations"]]:
        print("KIỂM THỬ CHÍNH XÁC: Phát hiện thiếu áo phản quang trong ZONE_02 thành công!")
    else:
        print(f"KIỂM THỬ CÓ LỖI: {roi_eval2}")

if __name__ == "__main__":
    test_roi_manager()
