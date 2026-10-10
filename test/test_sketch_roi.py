import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core_pipeline.roi_manager import ROIManagerOOP

def test_sketch_roi_manager():
    print("=== BẮT ĐẦU KIỂM THỬ ROI MANAGER DỰA TRÊN SƠ ĐỒ PHÁC THẢO ===")
    roi_manager = ROIManagerOOP()

    # Tạo Vùng 1
    roi_manager.add_zone(
        zone_id="ZONE_01",
        zone_name="Vùng 1",
        polygon_points_list=[(100, 100), (400, 100), (400, 400), (100, 400)],
        allowed_person_ids_list=["1", "2"] # Người 1 và Người 2 được phép
    )

    # Kiểm tra Người 1 (ID 1) vào Vùng 1 -> Phải AN TOÀN (hợp lệ)
    eval1 = roi_manager.evaluate_person_roi(track_id=1, bounding_box=[150, 100, 250, 300], ppe_detections_list=[])
    print(f"Kiểm tra Người 1 vào Vùng 1: {eval1}")
    if len(eval1[0]["zone_violations"]) == 0:
        print("KIỂM THỬ CHÍNH XÁC: Người 1 có trong danh sách -> Hợp lệ!")
    else:
        print(f"LỖI: {eval1}")

    # Kiểm tra Người 3 (ID 3) vào Vùng 1 -> Phải VI PHẠM (không có trong danh sách Vùng 1)
    eval3 = roi_manager.evaluate_person_roi(track_id=3, bounding_box=[150, 100, 250, 300], ppe_detections_list=[])
    print("--------------------------------------------------")
    print(f"Kiểm tra Người 3 vào Vùng 1: {eval3}")
    if len(eval3[0]["zone_violations"]) > 0 and eval3[0]["zone_violations"][0]["violation_type"] == "UNAUTHORIZED_INTRUSION":
        print("KIỂM THỬ CHÍNH XÁC: Người 3 KHÔNG có trong danh sách -> Vi phạm xâm nhập trái phép Vùng 1!")
    else:
        print(f"LỖI: {eval3}")

if __name__ == "__main__":
    test_sketch_roi_manager()
