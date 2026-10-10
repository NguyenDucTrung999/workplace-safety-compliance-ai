import sys
import os
import time

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core_pipeline.state_manager import SafetyStateManagerOOP

def test_state_manager():
    print("=== BẮT ĐẦU KIỂM THỬ SAFETY STATE MANAGER OOP ===")
    state_manager = SafetyStateManagerOOP(max_disappeared_seconds=10)
    
    current_time = time.time()
    
    # Khung hình 1: Track ID 1 xuất hiện với no helmet và no vest
    frame1_persons = [
        {
            "track_id": 1,
            "bounding_box": [100, 100, 200, 400],
            "confidence": 0.85,
            "ppe_detections": [
                {"class_id": 3, "class_name": "no helmet", "confidence": 0.90},
                {"class_id": 4, "class_name": "no vest", "confidence": 0.88}
            ]
        }
    ]
    
    # Giai đoạn 1: Duy trì 6 khung hình để kích hoạt bộ đếm xác nhận vi phạm
    for frame_idx in range(6):
        summary1, events1 = state_manager.update(frame1_persons, current_timestamp=current_time + (frame_idx * 0.03))
    print(f"Khung hình 1 - Số người theo vết: {summary1['total_tracked_persons']}")
    print(f"Khung hình 1 - Số vi phạm đang diễn ra: {summary1['total_active_violations']}")
    print(f"Khung hình 1 - Các sự kiện mới: {events1}")

    # Giai đoạn 2: Track ID 1 khắc phục vi phạm mũ (đội helmet on liên tục 6 frames) nhưng vẫn no vest
    frame2_persons = [
        {
            "track_id": 1,
            "bounding_box": [102, 100, 202, 400],
            "confidence": 0.86,
            "ppe_detections": [
                {"class_id": 1, "class_name": "helmet on", "confidence": 0.92},
                {"class_id": 4, "class_name": "no vest", "confidence": 0.87}
            ]
        }
    ]
    
    for frame_idx in range(6):
        summary2, events2 = state_manager.update(frame2_persons, current_timestamp=current_time + 2.0 + (frame_idx * 0.03))
    print("--------------------------------------------------")
    print(f"Khung hình 2 - Số người theo vết: {summary2['total_tracked_persons']}")
    print(f"Khung hình 2 - Số vi phạm đang diễn ra: {summary2['total_active_violations']}")
    print(f"Khung hình 2 - Các sự kiện mới: {events2}")
    
    active_violation_types = [v["violation_type"] for v in summary2["person_summaries"][0]["active_violations"]]
    if "no_helmet" not in active_violation_types and "no_vest" in active_violation_types:
        print("KIỂM THỬ CHÍNH XÁC: Vi phạm no helmet đã được giải quyết (RESOLVED), còn lại vi phạm no vest!")
    else:
        print(f"KIỂM THỬ CÓ LỖI: Các vi phạm đang có là {active_violation_types}")

    # Giai đoạn 3: Kiểm tra hành vi phone_usage và quy tắc không tự khắc phục
    frame3_persons = [
        {
            "track_id": 1,
            "bounding_box": [105, 100, 205, 400],
            "confidence": 0.88,
            "ppe_detections": [
                {"class_id": 1, "class_name": "helmet on", "confidence": 0.92},
                {"class_id": 5, "class_name": "vest", "confidence": 0.90},
                {"class_id": 9, "class_name": "phone_usage", "confidence": 0.85, "relative_y_center": 0.40}
            ]
        }
    ]
    for frame_idx in range(3):
        summary3, events3 = state_manager.update(frame3_persons, current_timestamp=current_time + 4.0 + (frame_idx * 0.03))

    # Công nhân cất điện thoại, không còn nhãn phone_usage trong 30 khung hình tiếp theo
    frame4_clean = [
        {
            "track_id": 1,
            "bounding_box": [105, 100, 205, 400],
            "confidence": 0.88,
            "ppe_detections": [
                {"class_id": 1, "class_name": "helmet on", "confidence": 0.92},
                {"class_id": 5, "class_name": "vest", "confidence": 0.90}
            ]
        }
    ]
    for frame_idx in range(30):
        summary4, events4 = state_manager.update(frame4_clean, current_timestamp=current_time + 6.0 + (frame_idx * 0.03))

    active_violations_stage3 = [v["violation_type"] for v in summary4["person_summaries"][0]["active_violations"]]
    if "phone_usage" in active_violations_stage3:
        print("KIỂM THỬ CHÍNH XÁC: Vi phạm phone_usage vẫn được lưu giữ và không tự khắc phục đúng yêu cầu!")
    else:
        print(f"KIỂM THỬ CÓ LỖI: phone_usage bị mất khỏi danh sách vi phạm: {active_violations_stage3}")

    # Giai đoạn 4: Kiểm thử Session Latching cho Thẻ nhân viên (id_card)
    # Track ID 2 xuất hiện đeo thẻ id_card trong 3 frames đầu tiên -> Trở thành compliant
    frame5_with_id = [
        {
            "track_id": 2,
            "bounding_box": [300, 100, 400, 400],
            "confidence": 0.90,
            "ppe_detections": [
                {"class_id": 1, "class_name": "helmet on", "confidence": 0.92},
                {"class_id": 5, "class_name": "vest", "confidence": 0.90},
                {"class_id": 8, "class_name": "id_card", "confidence": 0.85, "relative_y_center": 0.40}
            ]
        }
    ]
    for frame_idx in range(3):
        state_manager.update(frame5_with_id, current_timestamp=current_time + 8.0 + (frame_idx * 0.03))

    # Sau đó công nhân quay lưng / tháo thẻ trong 60 frames tiếp theo (~2 giây)
    frame5_without_id = [
        {
            "track_id": 2,
            "bounding_box": [300, 100, 400, 400],
            "confidence": 0.90,
            "ppe_detections": [
                {"class_id": 1, "class_name": "helmet on", "confidence": 0.92},
                {"class_id": 5, "class_name": "vest", "confidence": 0.90}
            ]
        }
    ]
    for frame_idx in range(60):
        summary6, _ = state_manager.update(frame5_without_id, current_timestamp=current_time + 10.0 + (frame_idx * 0.03))

    p2_summary = [p for p in summary6["person_summaries"] if p["track_id"] == 2][0]
    p2_violations = [v["violation_type"] for v in p2_summary["active_violations"]]
    if p2_summary["current_ppe_status"]["id_card"] == "compliant" and "no_id_card" not in p2_violations:
        print("KIỂM THỬ CHÍNH XÁC: Session Latching thẻ nhân viên (Cách 1) hoạt động hoàn hảo, không báo lỗi khi quay lưng/tháo thẻ!")
    else:
        print(f"KIỂM THỬ CÓ LỖI: Trạng thái thẻ của Track 2 là {p2_summary['current_ppe_status']['id_card']}, vi phạm: {p2_violations}")

if __name__ == "__main__":
    test_state_manager()
