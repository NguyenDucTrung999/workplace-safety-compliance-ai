import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core_pipeline.roi_manager import ROIManagerOOP

def test_roi_manager_assignments():
    print("=== BẮT ĐẦU KIỂM THỬ ROI MANAGER TRACK ASSIGNMENT ===")
    roi_manager = ROIManagerOOP()

    roi_manager.add_zone(
        zone_id="ZONE_A",
        zone_name="Khu Vuc A",
        zone_type="SAFE_ZONE",
        polygon_points_list=[(100, 100), (400, 100), (400, 400), (100, 400)]
    )

    roi_manager.add_zone(
        zone_id="ZONE_B",
        zone_name="Khu Vuc B",
        zone_type="SAFE_ZONE",
        polygon_points_list=[(500, 100), (900, 100), (900, 400), (500, 400)]
    )

    # Gán Track ID 1 CHỈ ĐƯỢC PHÉP ở ZONE_A
    roi_manager.assign_track_zones(track_id=1, allowed_zone_ids_list=["ZONE_A"])

    # Người Track ID 1 đứng ở ZONE_B (chân tại 600, 300) -> Phải phát vi phạm UNAUTHORIZED_ZONE_ACCESS!
    eval1 = roi_manager.evaluate_person_roi(track_id=1, bounding_box=[550, 100, 650, 300], ppe_detections_list=[])

    print("--------------------------------------------------")
    print(f"Kết quả kiểm tra Track ID 1 tại ZONE_B: {eval1}")
    
    if len(eval1) > 0 and "UNAUTHORIZED_ZONE_ACCESS" in [v["violation_type"] for v in eval1[0]["zone_violations"]]:
        print("KIỂM THỬ CHÍNH XÁC: Đã phát hiện công nhân ID 1 đi sai vùng phân công!")
    else:
        print(f"KIỂM THỬ CÓ LỖI: {eval1}")

if __name__ == "__main__":
    test_roi_manager_assignments()
